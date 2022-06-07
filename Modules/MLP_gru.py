import os
from select import select

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from tqdm import tqdm

from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup



class gruBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim ,n_layers=1, drop_prob=0.2):
        super(gruBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = out[:,-1]
        return out, h
    

class ourModel (nn.Module):
    def __init__(self):
        super(ourModel, self).__init__()
        self.gru_query_profile = gruBlock(input_dim=128, hidden_dim=200, n_layers=2, drop_prob=0.2)
        self.gru_query_dialogs = gruBlock(input_dim=100, hidden_dim=200, n_layers=2, drop_prob=0.2)
        self.relu= nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc1 = nn.Linear(200 * 2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def get_dialog_sent_masks(self, batch_dialogs_attention_mask):
        batch_dialogs_mask = []
        for i in range(batch_dialogs_attention_mask.shape[0]):
            batch_dialogs_mask.append(torch.where(torch.sum(batch_dialogs_attention_mask[i], dim=1) != 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()))
        batch_dialogs_mask = torch.stack(batch_dialogs_mask)
        return batch_dialogs_mask
    
    def forward(self, batch_query_emb, batch_profile_emb, batch_dialogs_emb, \
         batch_query_mask, batch_profile_mask, batch_dialogs_mask):
        
        batch_size, _, _ = batch_query_emb.shape

        query_profile_attn_mask_ = ~torch.bmm(batch_query_mask.unsqueeze(-1), batch_profile_mask.unsqueeze(1)).bool()
        query_profile_similarity_matrix = torch.bmm(batch_query_emb, batch_profile_emb.transpose(1,2))

        query_dialogs_attn_mask_ = ~torch.bmm(batch_query_mask.unsqueeze(-1), batch_dialogs_mask.unsqueeze(1)).bool()
        query_dialogs_similarity_matrix = torch.bmm(batch_query_emb, batch_dialogs_emb.transpose(1,2))

        h0_query_profile = self.gru_query_profile.init_hidden(batch_size).data
        out_query_profile, h_query_profile = self.gru_query_profile(query_profile_similarity_matrix, h0_query_profile)
        h0_query_dialogs = self.gru_query_dialogs.init_hidden(batch_size).data
        out_query_dialogs, h_query_dialogs = self.gru_query_dialogs(query_dialogs_similarity_matrix, h0_query_dialogs)

        features = torch.cat([out_query_profile ,out_query_dialogs], dim=1)
        features = self.relu(self.bn1(self.fc1(features)))
        output = self.fc2(features)
        output = self.sigmoid(output)

        return output.squeeze()

def train_epoch(train_dataloader, optimizer, model, tag):

    epoch_loss = []
    loss_fun = nn.BCELoss()
    for batch, labels in tqdm(train_dataloader):
        labels = labels.cuda()
        # print("labels", labels)
        batch = tuple(t.cuda() for t in batch)

        batch_profile_embed, batch_profile_attention_mask = batch[0], batch[1].float()
        batch_query_embed, batch_query_attention_mask = batch[2], batch[3].float()
        batch_dialogs_embed, batch_dialogs_attention_mask = batch[4], batch[5].float()
        # print("batch_dialogs_attnetion_mask", batch_dialogs_attention_mask.shape)
        batch_size, dr_dialog_num, max_len = batch_dialogs_embed.shape

        batch_dialogs_mask = model.get_dialog_sent_masks(batch_dialogs_attention_mask).float()
 
        if tag == "train":
            optimizer.zero_grad()
        logits = model(batch_query_embed, batch_profile_embed, batch_dialogs_embed, batch_query_attention_mask, batch_profile_attention_mask, batch_dialogs_mask)
        # print("logits", logits)
        loss = loss_fun(logits, labels)
        # print("loss", loss)
        print("loss", loss)
        print("logits", logits)
        print("labels", labels)
        if tag == "train":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            loss.backward()
            optimizer.step()
        epoch_loss.append(loss.item())
    return epoch_loss

def test_process(test_dataloader, model):

    logits_list = []
    num_ok = 0
    total_num = 0
    for batch, labels in tqdm(test_dataloader):
        batch = tuple(t.cuda() for t in batch)
        labels = labels.cuda()
        batch_profile_emb, batch_profile_attention_mask = batch[0], batch[1].float()
        batch_query_emb, batch_query_attention_mask = batch[2], batch[3].float()
        batch_dialogs_emb, batch_dialogs_attention_mask = batch[4], batch[5].float()
        batch_size, dr_dialog_num, max_len = batch_dialogs_emb.shape

        batch_dialogs_mask = model.get_dialog_sent_masks(batch_dialogs_attention_mask).float()


        # print("batch_dialog_emb: ", batch_dialogs_emb.shape)
        logits = model(batch_query_emb, batch_profile_emb, batch_dialogs_emb, batch_query_attention_mask, batch_profile_attention_mask, batch_dialogs_mask)
        print("logits: ", logits)
        total_num += 1
        # print("logits", logits.shape)
        print("labels", labels)
        logits_list.append(logits)
        if logits.float().argmax(dim=0) == 0:
            num_ok += 1
    print("Testing Accuracy: ", num_ok/total_num)
    return logits_list