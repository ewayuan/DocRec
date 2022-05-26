from random import sample
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class DoctorRecDataset(Dataset):
    def __init__(self, split, dataset, profile_ids, query_ids, dialogue_ids,
                 dr_dialog_sample=100, neg_sample=10, output=''):
        self.split = split
        self.dataset = dataset
        self.q_list = dataset.dialog_id.tolist() # query id - the same as dialogue id 
        self.dr_list = list(set(dataset.dr_id.tolist()))
        self.q_dr_match = dict(zip(dataset.dialog_id, dataset.dr_id))
        self.profile_ids = profile_ids
        self.q_ids = query_ids
        self.dialog_ids = dialogue_ids
        if split == "train":
            train_set = pd.read_csv('./dataset/train_mini.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id':str})
        elif split == "valid":
            train_set = pd.read_csv('./dataset/valid_mini.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id':str})
        self.most_common_drs = [dr for dr, _ in Counter(train_set.dr_id.tolist()).most_common()]
        self.train_q_dr_match = dict(zip(train_set.dialog_id, train_set.dr_id))
        del train_set
        self.dr_dialog_sample = dr_dialog_sample
        self.neg_sample = neg_sample
        self.output = output
        self.dr_feature = {}
        self.features = []
        self.labels = []
        for dr in tqdm(self.dr_list, desc='packing doctor features'):
            print("dr: ", dr)
            self.pack_dr_features(dr)
        self.pack_dataset()
        self.profile = []
        self.dialogue = []
        self.query = []
        print("self.features: ", len(self.features))

    def __getitem__(self, index):
        # self.features = [profile, dr_dialog_sample个doctor_dialogue , query]
        # profile:
        profile = self.features[index][0]
        profile_input_ids = self.features[index][0]["input_ids"]
        profile_token_type_ids = self.features[index][0]["token_type_ids"]
        profile_attention_mask = self.features[index][0]["attention_mask"]

        # query:
        query = self.features[index][-1]
        query_input_ids = self.features[index][-1]["input_ids"]
        query_token_type_ids = self.features[index][-1]["token_type_ids"]
        query_attention_mask = self.features[index][-1]["attention_mask"]

        # diagolues:
        diagolues = self.features[index][1 : 1+self.dr_dialog_sample]
        diagolues_input_ids = []
        diagolues_token_type_ids = []
        diagolues_attention_mask = []
        for d in diagolues:
            diagolues_input_ids.extend(d["input_ids"])
            diagolues_token_type_ids.extend(d["token_type_ids"])
            diagolues_attention_mask.extend(d["attention_mask"])

        return (torch.tensor(profile_input_ids), torch.tensor(profile_token_type_ids), torch.tensor(profile_attention_mask), \
               torch.tensor(query_input_ids), torch.tensor(query_token_type_ids), torch.tensor(query_attention_mask), \
               torch.stack(diagolues_input_ids), torch.stack(diagolues_token_type_ids), torch.stack(diagolues_attention_mask)),\
               torch.FloatTensor([self.labels[index]])[0]

    def __len__(self):
        return len(self.labels)

    def pack_dr_features(self, dr_id):
        # dr_feature: [profile, dialogue个dialogue]
        feature = []
        feature_profile = self.profile_ids[dr_id]
        feature.append(feature_profile)
        records = [dialog_id for (dialog_id, doctor_id) in self.train_q_dr_match.items() if doctor_id == dr_id]
        if len(records) > self.dr_dialog_sample:
            sample_records = sample(records, self.dr_dialog_sample)
            for idx in sample_records:
                feature.append(self.dialog_ids[idx])
        else:
            pad_size = self.dr_dialog_sample - len(records)
            for idx in records:
                feature.append(self.dialog_ids[idx])
            for i in range(pad_size):
                feature.append({"input_ids": torch.tensor([[0] *512]) ,\
                                "token_type_ids": torch.tensor([[0] *512]) ,\
                                "attention_mask": torch.tensor([[0] *512])  })
        self.dr_feature[dr_id] = feature

    def pack_dataset(self):
        if self.split == "test":
            test_dat = open(f'./{self.output}/test.dat', 'w', encoding='utf-8')
        for (q_idx, q) in enumerate(tqdm(self.q_list, desc=f'pack {self.split} dataset')):
            q_feature = self.q_ids[q]
            pos_dr = self.q_dr_match[q]
            pos_feature = self.dr_feature[pos_dr][:]
            pos_feature.append(q_feature)
            if self.split == 'test':
                print(f'# query {q_idx+1} {q} {pos_dr}', file=test_dat)
                print(f"1 'qid':{q_idx+1} # doctor: {pos_dr}", file=test_dat)
            self.features.append(pos_feature)
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
                neg_feature.append(q_feature)
                if self.split == 'test':
                    print(f"0 'qid':{q_idx+1} # doctor: {neg_dr}", file=test_dat)
                self.features.append(neg_feature)
                self.labels.append(0)
        if self.split == 'test':
            test_dat.close()
