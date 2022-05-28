from random import sample
from collections import Counter
from tkinter import dialog
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class DoctorRecDataset(Dataset):
    def __init__(self, split, dataset, profile, query, dialogue, 
                 dr_dialog_sample=100, neg_sample=10, embed_size=768, output=''):
        self.split = split
        self.dataset = dataset
        self.q_list = dataset.dialog_id.tolist() # query id - the same as dialogue id 
        self.dr_list = list(set(dataset.dr_id.tolist()))
        self.q_dr_match = dict(zip(dataset.dialog_id, dataset.dr_id))
        self.profile_emb = profile
        self.q_emb = query
        self.dialog_emb = dialogue
        if split == "train":
            train_set = pd.read_csv('./dataset/train_cleaned.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id':str})
        elif split == "valid":
            train_set = pd.read_csv('./dataset/valid_cleaned.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id':str})
        self.most_common_drs = [dr for dr, _ in Counter(train_set.dr_id.tolist()).most_common()]
        self.train_q_dr_match = dict(zip(train_set.dialog_id, train_set.dr_id))
        del train_set
        self.dr_dialog_sample = dr_dialog_sample
        self.neg_sample = neg_sample
        self.embed_size = embed_size
        self.dialog_max_len = 512
        self.output = output
        self.dr_feature = {}
        self.dr_masks = {}
        self.features = []
        self.masks = []
        self.labels = []
        for dr in tqdm(self.dr_list, desc='packing doctor features'):
            self.pack_dr_features(dr)
        self.pack_dataset()

    def __getitem__(self, index):

        # profile
        profile = self.features[index][0]
        profile_mask = self.masks[index][0]

        # query:
        query = self.features[index][-1]
        query_mask = self.masks[index][-1]

        # diagolues:
        dialog = self.features[index][1 : 1 + self.dr_dialog_sample]
        dialog_mask = self.masks[index][1 : 1 + self.dr_dialog_sample]

        return (torch.FloatTensor(profile), torch.FloatTensor(profile_mask), \
                torch.FloatTensor(query), torch.FloatTensor(query_mask), \
                torch.FloatTensor(np.array(dialog)), torch.FloatTensor(np.array(dialog_mask))), \
                torch.FloatTensor([self.labels[index]])[0]

    def __len__(self):
        return len(self.labels)

    def pack_dr_features(self, dr_id):
        feature = []
        masks = []
        feature_profile = self.profile_emb[dr_id]["embedding"]
        mask_profile = self.profile_emb[dr_id]["attention_mask"]
        feature.append(feature_profile)
        masks.append(mask_profile)
        records = [dialog_id for (dialog_id, doctor_id) in self.train_q_dr_match.items() if doctor_id == dr_id]
        if len(records) >= self.dr_dialog_sample:
            sample_records = sample(records, self.dr_dialog_sample)
            for idx in sample_records:
                feature.append(self.dialog_emb[idx]["embedding"])
                masks.append(self.dialog_emb[idx]["attention_mask"])
        else:
            pad_size = self.dr_dialog_sample - len(records)
            for idx in records:
                feature.append(self.dialog_emb[idx]["embedding"])
                masks.append(self.dialog_emb[idx]["attention_mask"])
            feature.extend([[0] * self.embed_size] * pad_size)
            masks.extend([[0] * self.dialog_max_len] * pad_size)
        self.dr_feature[dr_id] = feature
        self.dr_masks[dr_id] = masks
        return feature

    def pack_dataset(self):
        if self.split == "test":
            test_dat = open(f'./{self.output}/test.dat', 'w', encoding='utf-8')
        for (q_idx, q) in enumerate(tqdm(self.q_list, desc=f'pack {self.split} dataset')):
            q_feature = self.q_emb[q]["embedding"]
            q_mask = self.q_emb[q]["attention_mask"]
            pos_dr = self.q_dr_match[q]
            pos_feature = self.dr_feature[pos_dr][:]
            pos_mask = self.dr_masks[pos_dr][:]
            pos_feature.append(q_feature)
            pos_mask.append(q_mask)
            if self.split == 'test':
                print(f'# query {q_idx+1} {q} {pos_dr}', file=test_dat)
                print(f"1 'qid':{q_idx+1} # doctor: {pos_dr}", file=test_dat)
            self.features.append(pos_feature)
            self.masks.append(pos_mask)
            self.labels.append(1)

            # negtive sampling
            neg_drs = self.dr_list[:]
            neg_drs.remove(pos_dr)
            if self.split != 'test':
                neg_drs = sample(neg_drs, self.neg_sample)
            else:
                neg_drs = self.most_common_drs[:]
                neg_drs.remove(pos_dr)
                neg_drs = neg_drs[:100] # other top 100 doctors handling the most queries except current doctor
            for neg_dr in neg_drs:
                neg_feature = self.dr_feature[neg_dr][:]
                neg_mask = self.dr_masks[neg_dr][:]
                neg_feature.append(q_feature)
                neg_mask.append(q_mask)
                if self.split == 'test':
                    print(f"0 'qid':{q_idx+1} # doctor: {neg_dr}", file=test_dat)
                self.features.append(neg_feature)
                self.masks.append(neg_mask)
                self.labels.append(0)
        if self.split == 'test':
            test_dat.close()