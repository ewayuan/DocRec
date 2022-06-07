import os

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from tqdm import tqdm

from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

import logging
import GPUtil

class TransformerBlock(nn.Module):

    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, Q, K, V, mask=None, dropout=None, episilon=1e-8):
        '''
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        if mask is not None:
            Q_K = Q_K.masked_fill_(mask, -1e9)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)
        if dropout is not None:
            Q_K_score = dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            X = self.linear2(self.relu(self.linear1(X))) + X

        return X

class cnnBlock(nn.Module):
    def __init__(self):
        super(cnnBlock, self).__init__()
        self.cnn_2d_query_profile_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_query_profile_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_query_profile_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.maxpooling_query_profile_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.affine_query_profile = nn.Linear(in_features=15625, out_features=2000)
        self.relu = nn.ReLU()

        self.cnn_2d_query_dialogs_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_query_dialogs_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_query_dialogs_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.maxpooling_query_dialogs_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.affine_query_dialogs = nn.Linear(in_features=12125, out_features=2000)
        
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.fc1 = nn.Linear(in_features =2000 * 4, out_features = 768)
        self.fc2 = nn.Linear(768, 1)

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.cnn_2d_query_profile_1.weight)
        init.xavier_normal_(self.cnn_2d_query_profile_2.weight)
        init.xavier_normal_(self.affine_query_profile.weight)

        init.xavier_normal_(self.cnn_2d_query_dialogs_1.weight)
        init.xavier_normal_(self.cnn_2d_query_dialogs_2.weight)
        init.xavier_normal_(self.affine_query_dialogs.weight)
        
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def cnn_query_profile(self, matrix):

        matrix = matrix.unsqueeze(1)
        # print("cnn_query_profile matrix: ", matrix)
        # print("self.relu(self.cnn_2d_query_profile_1(matrix)): ", self.relu(self.cnn_2d_query_profile_1(matrix)).shape)
        Z = self.relu(self.cnn_2d_query_profile_1(matrix))
        # print("cnn_query_profile Z1: ", Z)
        Z = self.maxpooling_query_profile_1(Z)
        # print("cnn_query_profile Z2: ", Z)
        Z = Z.view(Z.size(0), -1)
 
        output_V = self.affine_query_profile(Z)

        return output_V

    def cnn_query_dialogs(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.relu(self.cnn_2d_query_dialogs_1(matrix))
        # print("cnn_query_dialogs Z1: ", Z)
        Z = self.maxpooling_query_dialogs_1(Z)
        # print("cnn_query_dialogs Z2: ", Z)
        Z = Z.view(Z.size(0), -1)

        output_V = self.affine_query_dialogs(Z)
        return output_V

    def forward(self, query_profile_similarity_matrix, query_dialogs_similarity_matrix, \
                query_profile_interaction_simialrity_matrix, query_dialogs_interaction_simialrity_matrix):

        query_profile_V = self.cnn_query_profile(query_profile_similarity_matrix)
        query_profile_interaction_V = self.cnn_query_profile(query_profile_interaction_simialrity_matrix)

        query_dialogs_V = self.cnn_query_dialogs(query_dialogs_similarity_matrix)
        query_dialogs_interaction_V = self.cnn_query_dialogs(query_dialogs_interaction_simialrity_matrix)

        # print("query_profile_V: ", query_profile_V)
        # print("query_profile_interaction_V: ", query_profile_interaction_V)
        # print("query_dialogs_V: ", query_dialogs_V)
        # print("query_dialogs_interaction_V: ", query_dialogs_interaction_V)
        features = torch.cat([query_profile_V, query_dialogs_V, query_profile_interaction_V, query_dialogs_interaction_V], dim=-1)
        features = self.relu(self.bn1(self.fc1(features)))
        output = self.fc2(features)
        output = self.sigmoid(output)
        return output.squeeze()

class ourModel (nn.Module):
    def __init__(self):
        super(ourModel, self).__init__()
        self.cnn_block = cnnBlock()
        self.embed_dim = 768
        self.query_transformer  = TransformerBlock(input_size=self.embed_dim)
        self.profile_transformer = TransformerBlock(input_size=self.embed_dim)
        self.dialogs_transformer = TransformerBlock(input_size=self.embed_dim)

    def get_dialog_sent_masks(self, batch_dialogs_attention_mask):
        batch_dialogs_mask = []
        for i in range(batch_dialogs_attention_mask.shape[0]):
            batch_dialogs_mask.append(torch.where(torch.sum(batch_dialogs_attention_mask[i], dim=1) != 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()))
        batch_dialogs_mask = torch.stack(batch_dialogs_mask)
        return batch_dialogs_mask
    
    def forward(self, batch_query_emb, batch_profile_emb, batch_dialogs_emb, \
         batch_query_mask, batch_profile_mask, batch_dialogs_mask):

        batch_query_emb = batch_query_emb.masked_fill(batch_query_mask.unsqueeze(-1)==0, -1e9)
        batch_profile_emb = batch_profile_emb.masked_fill(batch_profile_mask.unsqueeze(-1)==0, -1e9)
        batch_dialogs_emb = batch_dialogs_emb.masked_fill(batch_dialogs_mask.unsqueeze(-1)==0, -1e9)


        batch_query_emb_norm = torch.nn.functional.normalize(batch_query_emb, dim=-1) 
        batch_profile_emb_norm = torch.nn.functional.normalize(batch_profile_emb, dim=-1) 
        batch_dialogs_emb_norm = torch.nn.functional.normalize(batch_dialogs_emb, dim=-1) 

        query_profile_similarity_matrix = torch.bmm(batch_query_emb_norm, batch_profile_emb_norm.transpose(1,2))
        query_dialogs_similarity_matrix = torch.bmm(batch_query_emb_norm, batch_dialogs_emb_norm.transpose(1,2))

        # Interaction
        query_profile_attn_mask_ = ~torch.bmm(batch_query_mask.unsqueeze(-1), batch_profile_mask.unsqueeze(1)).bool()
        query_dialogs_attn_mask_ = ~torch.bmm(batch_query_mask.unsqueeze(-1), batch_dialogs_mask.unsqueeze(1)).bool()

        query_add_profile_attn = self.query_transformer(batch_query_emb_norm, batch_profile_emb_norm, batch_profile_emb_norm, mask=~query_profile_attn_mask_)
        profile_add_query_attn = self.profile_transformer(batch_profile_emb_norm, batch_query_emb_norm, batch_query_emb_norm, mask=~query_profile_attn_mask_.transpose(1,2))

        query_add_dialogs_attn = self.query_transformer(batch_query_emb_norm, batch_dialogs_emb_norm, batch_dialogs_emb_norm, mask=~query_dialogs_attn_mask_)
        dialogs_add_query_attn = self.dialogs_transformer(batch_dialogs_emb_norm, batch_query_emb_norm, batch_query_emb_norm, mask=~query_dialogs_attn_mask_.transpose(1,2))

        query_profile_interaction_simialrity_matrix = torch.bmm(query_add_profile_attn, profile_add_query_attn.transpose(1,2))

        query_dialogs_interaction_simialrity_matrix = torch.bmm(query_add_dialogs_attn, dialogs_add_query_attn.transpose(1,2))

        # print("batch_query_emb: ", batch_query_emb)
        # print("batch_profile_emb: ", batch_profile_emb)
        # print("batch_dialogs_emb: ", batch_dialogs_emb)
        # print("batch_query_emb_norm: ", batch_query_emb_norm)
        # print("batch_profile_emb_norm: ", batch_profile_emb_norm)
        # print("batch_dialogs_emb_norm: ", batch_dialogs_emb_norm)

        # print("query_profile_similarity_matrix: ", query_profile_similarity_matrix)
        # print("query_dialogs_similarity_matrix: ", query_dialogs_similarity_matrix)
        # print("query_profile_interaction_simialrity_matrix: ", query_profile_interaction_simialrity_matrix)
        # print("query_dialogs_interaction_simialrity_matrix: ", query_dialogs_interaction_simialrity_matrix)

        output = self.cnn_block(query_profile_similarity_matrix, query_dialogs_similarity_matrix, \
                                query_profile_interaction_simialrity_matrix, query_dialogs_interaction_simialrity_matrix)

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
        
        loss = loss_fun(logits, labels)
        # print("loss", loss)
        # print("loss", loss)
        # print("logits", logits)
        # print("labels", labels)
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
        # print("logits: ", logits)
        total_num += 1
        # print("logits", logits.shape)
        # print("labels", labels)
        logits_list.append(logits)
        if logits.float().argmax(dim=0) == 0:
            num_ok += 1
    print("Testing Accuracy: ", num_ok/total_num)
    return logits_list