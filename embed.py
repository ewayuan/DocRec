import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from utils import util

parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=2021, type=int)
parser.add_argument('-model', default='bert', type=str)
args = parser.parse_args()

# set you available gpus
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
np.random.seed(args.seed)
if args.model == 'bert':
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    print('gpu num: ', n_gpu)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def convert_sent_to_embed(model, tokenizer, device, id_content, max_len, output_path, type):
    con_emb_dict = {}
    for idx, content in tqdm(id_content.items(), desc=output_path):
        if pd.isna(content):
            continue
        encoded = tokenizer.encode_plus(
                text=content,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = max_len,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                truncation=True,
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
        ).to(device)
        output = model(**encoded)[0]
        mask = encoded['attention_mask'].squeeze()
        output = output.masked_fill(mask.unsqueeze(-1) == 0, 5e-5).squeeze()
        if type == "dialogues":
            mean_embed = (output * mask.unsqueeze(-1)).sum(dim=0) / mask.sum(dim=0)
            con_emb_dict[idx] = {"embedding": mean_embed.detach().cpu().numpy(), "attention_mask": mask.detach().cpu().numpy()}
        else:
            con_emb_dict[idx] = {"embedding": output.detach().cpu().numpy(), "attention_mask": mask.detach().cpu().numpy()}
    util.save_pickle(con_emb_dict, output_path)


def main():
    df = pd.read_csv(f'./dataset/embed_cleaned.csv', delimiter='\t', encoding='utf-8')
    df = df[['dr_id', 'dialog_id', 'q', 'parsed_dialog']]
    id_profile = {}
    with open(f'./dataset/dr_profile.jsonl', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            id_profile[line['id']] = line['goodat'] # use goodat as doctor profile
    id_q = dict(zip(df.dialog_id.tolist(), df.q.tolist()))
    id_dialog = dict(zip(df.dialog_id.tolist(), df.parsed_dialog.tolist()))
    
    if args.model == 'bert':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('./mc_bert_base/')
        model = BertModel.from_pretrained('./mc_bert_base/')
        model = model.to(device)
        cleaned_path = './cleaned'
        if not os.path.exists(cleaned_path):
            os.makedirs(cleaned_path)

        convert_sent_to_embed(model, tokenizer, device, id_profile, 128, f'{cleaned_path}/profile_embeds.pkl', "profile")
        convert_sent_to_embed(model, tokenizer, device, id_q, 128, f'{cleaned_path}/q_embeds.pkl', "query")
        convert_sent_to_embed(model, tokenizer, device, id_dialog, 512, f'{cleaned_path}/dialog_embeds.pkl', "dialogues")
if __name__ == '__main__':
    main()
