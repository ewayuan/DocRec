import os

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup


import torch.nn.init as init

import logging

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
        self.cnn_2d_context_response_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_context_response_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_context_response_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.affine_context_response = nn.Linear(in_features=6*62*1, out_features=200)
        self.relu = nn.ReLU()

        self.cnn_2d_persona_response_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_persona_response_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_persona_response_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.affine_persona_response = nn.Linear(in_features=6*56*1, out_features=200)
        self.affine_out = nn.Linear(in_features=200*6, out_features=1)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.cnn_2d_context_response_1.weight)
        init.xavier_normal_(self.cnn_2d_context_response_2.weight)
        init.xavier_normal_(self.affine_context_response.weight)

        init.xavier_normal_(self.cnn_2d_persona_response_1.weight)
        init.xavier_normal_(self.cnn_2d_persona_response_2.weight)
        init.xavier_normal_(self.affine_persona_response.weight)

        init.xavier_normal_(self.affine_out.weight)

    def cnn_contxt_response(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.cnn_2d_context_response_1(matrix)
        Z = self.maxpooling_context_response_1(Z)
        Z = self.relu(Z)

        Z = self.cnn_2d_context_response_2(Z)
        Z = self.maxpooling_context_response_1(Z)
        Z = self.relu(Z)

        Z = Z.view(Z.size(0), -1)

        output_V = self.affine_context_response(Z)

        return output_V

    def cnn_persona_response(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.cnn_2d_persona_response_1(matrix)
        Z = self.maxpooling_persona_response_1(Z)
        Z = self.relu(Z)

        Z = self.cnn_2d_persona_response_2(Z)
        Z = self.maxpooling_persona_response_1(Z)
        Z = self.relu(Z)

        Z = Z.view(Z.size(0), -1)

        output_V = self.affine_persona_response(Z)

        return output_V

    def forward(self, context_response_similarity_matrix, persona_response_similarity_matrix):

        # context_response_attn_V = self.cnn_contxt_response(context_response_attn_similarity_matrix)
        context_response_V = self.cnn_contxt_response(context_response_similarity_matrix)
        # persona_response_attn_V = self.cnn_persona_response(persona_response_attn_similarity_matrix)
        persona_response_V = self.cnn_persona_response(persona_response_similarity_matrix)

        all_concat = torch.cat([context_response_V, persona_response_V], dim=-1)
        matching_output = self.affine_out(all_concat)

        return matching_output.squeeze()

class ourModel (nn.Module):
    def __init__(self):
        super(ourModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./mc_bert_base/')
        self.bert_model =  BertModel.from_pretrained('./mc_bert_base/', output_hidden_states=False)
        self.embed_dim = 768
        self.response_len = 32
        self.context_len = 256
        self.persona_len = 231
        self.context_transformer  = TransformerBlock(input_size=self.embed_dim)
        self.response_transformer = TransformerBlock(input_size=self.embed_dim)
        self.persona_transformer = TransformerBlock(input_size=self.embed_dim)
        self.cnn_block = cnnBlock()

    def process_dialogues(self, batch_diagolues_input_ids, batch_diagolues_token_type_ids, batch_diagolues_attention_mask):

        batch_size, dr_dialog_num, max_len = batch_diagolues_input_ids.shape
        print("dr_dialog_num: ",dr_dialog_num)

        for dialog_idx in range(dr_dialog_num):
            print("================", dialog_idx, "================")
            print("batch_diagolues_input_ids[:,dialog_idx,:]: ", batch_diagolues_input_ids[:,dialog_idx,:].shape)
            print("batch_diagolues_attention_mask[:,dialog_idx,:]: ", batch_diagolues_attention_mask[:,dialog_idx,:].shape)
            print("batch_diagolues_token_type_ids[:,dialog_idx,:]: ", batch_diagolues_token_type_ids[:,dialog_idx,:].shape)

            batch_dialog = {"input_ids": batch_diagolues_input_ids[:,dialog_idx,:], \
                          "attention_mask": batch_diagolues_attention_mask[:,dialog_idx,:], \
                          "token_type_ids": batch_diagolues_token_type_ids[:,dialog_idx,:]}
            print("batch_dialog: ", batch_dialog)
            # take avery of last hidden layer: 1 * emb_size
            output_cur_dialog_emb = self.bert_model(**batch_dialog)[0]
            print("output_cur_dialog_emb:", output_cur_dialog_emb.shape)
        cur_batch_dr_dialog_emb.append(torch.mean(output_cur_dialog_emb, dim=0))


    def train_epoch(self, train_dataloader, optimizer, model):
        for batch, labels in tqdm(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            labels = labels.cuda()
            batch_profile_input_ids, batch_profile_token_type_ids, batch_profile_attention_mask = batch[0].squeeze(), batch[1].squeeze(), batch[2].squeeze().float()
            batch_query_input_ids, batch_query_token_type_ids, batch_query_attention_mask =  batch[3].squeeze(), batch[4].squeeze(), batch[5].squeeze().float()
            batch_diagolues_input_ids, batch_diagolues_token_type_ids, batch_diagolues_attention_mask = batch[6], batch[7], batch[8]
            print("batch_profile_input_ids: ", batch_profile_input_ids.shape)
            print("batch_profile_token_type_ids: ", batch_profile_token_type_ids.shape)
            print("batch_profile_attention_mask: ", batch_profile_attention_mask.shape)

            print("batch_query_input_ids: ", batch_query_input_ids.shape)
            print("batch_query_token_type_ids: ", batch_query_token_type_ids.shape)
            print("batch_query_attention_mask: ", batch_query_attention_mask.shape)

            print("batch_diagolues_input_ids: ", batch_diagolues_input_ids.shape)
            print("batch_diagolues_token_type_ids: ", batch_diagolues_token_type_ids.shape)
            print("batch_diagolues_attention_mask: ", batch_diagolues_attention_mask.shape)

            print("labels: ", labels)
            print("labels: ", labels.shape)

            # batch_profile = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            batch_profile = {"input_ids": batch_profile_input_ids, "attention_mask": batch_profile_attention_mask, "token_type_ids": batch_profile_token_type_ids}
            batch_query = {"input_ids": batch_query_input_ids, "attention_mask": batch_query_attention_mask, "token_type_ids": batch_query_token_type_ids}


            output_profile_emb = self.bert_model(**batch_profile)[0] # batch_size, max_len, emb_size, take last_hidden_status
            output_query_emb = self.bert_model(**batch_query)[0] # batch_size, max_len, emb_size, take last_hidden_status

            print("output_profile_emb: ", output_profile_emb.shape)
            print("output_query_emb: ", output_query_emb.shape)

            batch_size, dr_dialog_num, max_len = batch_diagolues_input_ids.shape
            os.system("nvidia-smi")
            batch_dialog_emb = self.process_dialogues(batch_diagolues_input_ids, batch_diagolues_token_type_ids, batch_diagolues_attention_mask)
            print("batch_dialog_emb: ", batch_dialog_emb.shape)

            logits = model (output_profile_emb, output_query_emb, batch_dialog_emb)
            optimizer.zero_grad()
            pred_scores = model(features, args.dr_dialog_sample)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    def cal_topic_embedding(self):
        all_topic_frequent_words_prob_embeddings = []
        for frequent_words_probabilities, coherence in self.dialog_data_topic:
            cur_topic_frequent_words_prob_embeddings = []
            cur_total_probability = 0
            for probability, word in frequent_words_probabilities:
                cur_total_probability += probability
            for probability, word in frequent_words_probabilities:
                word_encoded = self.tokenizer.encode_plus(word, return_tensors="pt", add_special_tokens=False)
                with torch.no_grad():
                    outputs = self.bert_model(**word_encoded)
                    word_embedding = outputs[0]
                    cur_word_embedding_with_prob = (probability/cur_total_probability * word_embedding.squeeze(0)).mean(dim=0)
                cur_topic_frequent_words_prob_embeddings.append(cur_word_embedding_with_prob)
            all_topic_frequent_words_prob_embeddings.append(torch.mean(torch.stack(cur_topic_frequent_words_prob_embeddings), dim=0))
        return all_topic_frequent_words_prob_embeddings

    def forward(self, batch_context_emb, batch_response_emb, batch_persona_emb, \
         batch_context_mask, batch_response_mask, batch_persona_mask):

        context_response_attn_mask_ = ~torch.bmm(batch_context_mask.unsqueeze(-1), batch_response_mask.unsqueeze(1)).bool()
        context_response_similarity_matrix = torch.bmm(batch_context_emb, batch_response_emb.transpose(1,2))
        context_response_similarity_matrix = context_response_similarity_matrix.masked_fill_(context_response_attn_mask_, 0)

        # persona_response_attn_similarity_matrix = torch.bmm(persona_attn_output, response_attn_output.transpose(1,2))
        # persona_response_attn_similarity_matrix = persona_response_attn_similarity_matrix.masked_fill_(persona_response_attn_mask, 0)

        persona_response_attn_mask_ = ~torch.bmm(batch_persona_mask.unsqueeze(-1), batch_response_mask.unsqueeze(1)).bool()
        persona_response_similarity_matrix = torch.bmm(batch_persona_emb, batch_response_emb.transpose(1,2))
        persona_response_similarity_matrix = persona_response_similarity_matrix.masked_fill_(persona_response_attn_mask_, 0)




        output = self.cnn_block(context_response_similarity_matrix, persona_response_similarity_matrix)

        return output.squeeze()
