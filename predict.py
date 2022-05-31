import argparse
import json
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time

from Modules.MLP import ourModel, test_process
from utils.dataset import DoctorRecDataset
from utils.config import init_opts, train_opts, eval_opts, multihead_att_opts
from utils.util import load_pickle, save_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default="med-bert", type=str)
parser.add_argument('--cleaned_path', default='./cleaned', type=str)


parser.add_argument('--dr_dialog_sample', default=100, type=int)
parser.add_argument('--neg_sample', default=10, type=int)
parser.add_argument('--batch_size', default=101, type=int)
parser.add_argument('--lr', default=2e-5, type=int)
parser.add_argument('--patience', default=7, type=int)
parser.add_argument('--output_dir', default="saved_model", type=str)
parser.add_argument('--epoch_num', default=10, type=int)

parser.add_argument('--eval_model', default="model_5.pt", type=str)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.set_device(args.gpu)

print(f'Prediction: randome seed {args.seed}, experiment name: {args.name}, run on gpu {args.gpu}')

def main():
    print(f'{args.name} start!')
    print(f'Loadding embeddings from {args.cleaned_path}...')
    profile_embeds = load_pickle(f'{args.cleaned_path}/profile_embeds.pkl')
    print("Loaded profile embeddings")
    query_embeds = load_pickle(f'{args.cleaned_path}/q_embeds.pkl')
    print("Loaded query embeddings")
    dialogue_embeds = load_pickle(f'{args.cleaned_path}/dialog_embeds.pkl')
    print("Loaded dialogue embeddings")
    end  = time.time()
    
    print('Building test dataset and dataloader...')
    test_set = pd.read_csv(f'./dataset/test_cleaned.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    test_dataset = DoctorRecDataset(
        'test', test_set, profile_embeds, query_embeds, dialogue_embeds,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    print("test_dataset length: ", len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    del test_set, test_dataset, profile_embeds, query_embeds, dialogue_embeds,
    print('Done')
    
    model = ourModel().cuda()
    model_path = f'{args.output_dir}/ckpt/{args.eval_model}'
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = model.cuda()
    
    print(f'{args.name} start prediction...')


    with open(f'{args.output_dir}/test_{args.eval_model}_score.txt', 'w', encoding='utf-8') as score:
        model.eval()
        with torch.no_grad():
            pred_scores = test_process(test_dataloader, model)
            for pred_score in pred_scores:
                print(pred_score.cpu().detach().numpy().tolist(), file = score)

if __name__ == '__main__':
    main()