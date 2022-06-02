import os

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from tqdm import tqdm

from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

import logging

class simpleBlock(nn.Module):
    def __init__(self):
        super(simpleBlock, self).__init__()
        self.affine_out = nn.Linear(in_features=1536, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.affine_out.weight)

    def forward(self, query_emb, profile_emb, query_mask, profile_mask):

        query_V = torch.mean(query_emb, dim=1)
        profile_V = torch.mean(profile_emb, dim=1)
        all_concat = torch.cat([query_V, profile_V], dim=-1)
        matching_output = self.affine_out(all_concat)
        output = self.sigmoid(matching_output)
        return output.squeeze()

class ourModel (nn.Module):
    def __init__(self):
        super(ourModel, self).__init__()
        self.simpleBlock = simpleBlock()

    def get_dialog_sent_masks(self, batch_dialogs_attention_mask):
        batch_dialogs_mask = []
        for i in range(batch_dialogs_attention_mask.shape[0]):
            batch_dialogs_mask.append(torch.where(torch.sum(batch_dialogs_attention_mask[i], dim=1) != 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()))
        batch_dialogs_mask = torch.stack(batch_dialogs_mask)
        return batch_dialogs_mask
    
    def forward(self, batch_query_emb, batch_profile_emb, batch_dialogs_emb, \
         batch_query_mask, batch_profile_mask, batch_dialogs_mask):
        output = self.simpleBlock(batch_query_emb, batch_profile_emb, batch_query_mask, batch_profile_mask)
        return output.squeeze()

def train_epoch(train_dataloader, optimizer, model, tag):
    epoch_loss = []
    loss_fun = nn.BCELoss()
    for batch, labels in tqdm(train_dataloader):
        labels = labels.cuda()
        batch = tuple(t.cuda() for t in batch)

        batch_profile_embed, batch_profile_attention_mask = batch[0], batch[1].float()
        batch_query_embed, batch_query_attention_mask = batch[2], batch[3].float()
        batch_dialogs_embed, batch_dialogs_attention_mask = batch[4], batch[5].float()
        batch_size, dr_dialog_num, max_len = batch_dialogs_embed.shape

        batch_dialogs_mask = model.get_dialog_sent_masks(batch_dialogs_attention_mask).float()
 
        if tag == "train":
            optimizer.zero_grad()
        logits = model(batch_query_embed, batch_profile_embed, batch_dialogs_embed, batch_query_attention_mask, batch_profile_attention_mask, batch_dialogs_mask)
        #print("labels", labels)
        #print("logits", logits)
        loss = loss_fun(logits, labels)
        #print("loss", loss)

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

        logits = model(batch_query_emb, batch_profile_emb, batch_dialogs_emb, batch_query_attention_mask, batch_profile_attention_mask, batch_dialogs_mask)
        print("logits: ", logits)
        total_num += 1

        logits_list.append(logits)
        if logits.float().argmax(dim=0) == 0:
            num_ok += 1
    print("Testing Accuracy: ", num_ok/total_num)
    return logits_list