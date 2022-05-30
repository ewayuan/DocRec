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

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        '''
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)
        V_att = Q_K_score.bmm(V)
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output

class cnnBlock(nn.Module):
    def __init__(self):
        super(cnnBlock, self).__init__()
        self.cnn_2d_query_profile_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_query_profile_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_query_profile_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.affine_query_profile = nn.Linear(in_features=900, out_features=200)
        self.relu = nn.ReLU()

        self.cnn_2d_query_dialogs_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1))
        self.cnn_2d_query_dialogs_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1))
        self.maxpooling_query_dialogs_1 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.affine_query_dialogs = nn.Linear(in_features=12800, out_features=200)
        self.affine_out = nn.Linear(in_features=400, out_features=1)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.cnn_2d_query_profile_1.weight)
        init.xavier_normal_(self.cnn_2d_query_profile_2.weight)
        init.xavier_normal_(self.affine_query_profile.weight)

        init.xavier_normal_(self.cnn_2d_query_dialogs_1.weight)
        init.xavier_normal_(self.cnn_2d_query_dialogs_2.weight)
        init.xavier_normal_(self.affine_query_dialogs.weight)

        init.xavier_normal_(self.affine_out.weight)

    def cnn_query_profile(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.cnn_2d_query_profile_1(matrix)
        Z = self.maxpooling_query_profile_1(Z)
        Z = self.relu(Z)

        Z = self.cnn_2d_query_profile_2(Z)
        Z = self.maxpooling_query_profile_1(Z)
        Z = self.relu(Z)

        Z = Z.view(Z.size(0), -1)
 
        output_V = self.affine_query_profile(Z)

        return output_V

    def cnn_query_dialogs(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.cnn_2d_query_dialogs_1(matrix)
        Z = self.maxpooling_query_dialogs_1(Z)
        Z = self.relu(Z)

        Z = self.cnn_2d_query_dialogs_2(Z)
        Z = self.maxpooling_query_dialogs_1(Z)
        Z = self.relu(Z)

        Z = Z.view(Z.size(0), -1)

        output_V = self.affine_query_dialogs(Z)

        return output_V

    def forward(self, query_profile_similarity_matrix, query_dialogs_similarity_matrix):

        query_profile_V = self.cnn_query_profile(query_profile_similarity_matrix)
        # print("query_profile_V", query_profile_V.shape)

        query_dialogs_V = self.cnn_query_dialogs(query_dialogs_similarity_matrix)
        # print("query_dialogs_V", query_dialogs_V.shape)

        all_concat = torch.cat([query_profile_V, query_dialogs_V], dim=-1)
        matching_output = self.affine_out(all_concat)

        return matching_output.squeeze()

class ourModel (nn.Module):
    def __init__(self):
        super(ourModel, self).__init__()
        self.cnn_block = cnnBlock()
        
    def get_dialog_sent_masks(self, batch_dialogs_attention_mask):
        batch_dialogs_mask = []
        for i in range(batch_dialogs_attention_mask.shape[0]):
            batch_dialogs_mask.append(torch.where(torch.sum(batch_dialogs_attention_mask[i], dim=1) != 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()))
        batch_dialogs_mask = torch.stack(batch_dialogs_mask)
        return batch_dialogs_mask
    
    def forward(self, batch_query_emb, batch_profile_emb, batch_dialogs_emb, \
         batch_query_mask, batch_profile_mask, batch_dialogs_mask):

        #  print("query", batch_query_mask.shape)
        #  print("profile", batch_profile_mask.shape) 
        #  print("dialogs", batch_dialogs_mask.shape) 
        query_profile_attn_mask_ = ~torch.bmm(batch_query_mask.unsqueeze(-1), batch_profile_mask.unsqueeze(1)).bool()
        query_profile_similarity_matrix = torch.bmm(batch_query_emb, batch_profile_emb.transpose(1,2))
        query_profile_similarity_matrix = query_profile_similarity_matrix.masked_fill_(query_profile_attn_mask_, 0) # batch_size * query_max_len * profile_max_len


        # query_dialogs_attn_similarity_matrix = torch.bmm(persona_attn_output, response_attn_output.transpose(1,2))
        # query_dialogs_attn_similarity_matrix = query_dialogs_attn_similarity_matrix.masked_fill_(query_dialogs_attn_mask, 0)

        query_dialogs_attn_mask_ = ~torch.bmm(batch_query_mask.unsqueeze(-1), batch_dialogs_mask.unsqueeze(1)).bool()
        query_dialogs_similarity_matrix = torch.bmm(batch_query_emb, batch_dialogs_emb.transpose(1,2))
        query_dialogs_similarity_matrix = query_dialogs_similarity_matrix.masked_fill_(query_dialogs_attn_mask_, 0)
        #  print("query_profile_similarity_matrix", query_profile_similarity_matrix.shape)
        # print("query_dialogs_similarity_matrix", query_dialogs_similarity_matrix.shape)

        
        output = self.cnn_block(query_profile_similarity_matrix, query_dialogs_similarity_matrix)
        return output.squeeze()

def train_epoch(train_dataloader, optimizer, model, tag):

    epoch_loss = []

    for batch, labels in tqdm(train_dataloader):
        labels = labels.cuda()
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
        loss = F.binary_cross_entropy_with_logits(logits, labels)
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

    for batch, labels in tqdm(test_dataloader):
        batch = tuple(t.cuda() for t in batch)
        labels = labels.cuda()
        batch_profile_emb, batch_profile_attention_mask = batch[0], batch[1].float()
        batch_query_emb, batch_query_attention_mask = batch[2], batch[3].float()
        batch_dialogs_emb, batch_dialogs_attention_mask = batch[4], batch[5].float()
        batch_size, dr_dialog_num, max_len = batch_dialogs_emb.shape

        batch_dialogs_mask = model.get_dialog_sent_masks(batch_dialogs_attention_mask).float()
       
        batch_size, dr_dialog_num, max_len = batch_dialogs_input_ids.shape
        batch_dialogs_emb = self.process_dialogues(batch_dialogs_input_ids, batch_dialogs_token_type_ids, batch_dialogs_attention_mask)
        # print("batch_dialog_emb: ", batch_dialogs_emb.shape)
        logits = model(batch_query_emb, batch_profile_emb, batch_dialogs_emb, batch_query_attention_mask, batch_profile_attention_mask, batch_dialogs_mask)
        logits_list.append(logits)
    return logits_list