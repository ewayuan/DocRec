import argparse
import json
from collections import Counter
import random
import os 

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
from Modules.MLP_interaction import ourModel
from Modules.MLP_interaction import train_epoch
from utils.dataset import DoctorRecDataset
from utils.EarlyStopping import EarlyStopping
from utils.loss import weighted_class_bceloss
from utils.config import init_opts, train_opts, multihead_att_opts

from utils.util import save_pickle, load_pickle
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup



parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--n_gpu', default=4, type=int)

parser.add_argument('--name', default="med-bert", type=str)
parser.add_argument('--cleaned_path', default='./cleaned', type=str)

parser.add_argument('--dr_dialog_sample', default=100, type=int)
parser.add_argument('--neg_sample', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=2e-5, type=int)
parser.add_argument('--patience', default=7, type=int)
parser.add_argument('--output_dir', default="saved_model", type=str)
parser.add_argument('--epoch_num', default=10, type=int)


args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.set_device(args.gpu)

print(f'Training: randome seed {args.seed}, experiment name: {args.name}, run on gpu {args.gpu}')

def train_model(model, train_dataloader, val_dataloader):
    print(f'{args.name} start training...')
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, threshold=1e-4, min_lr=1e-5)
    early_stopping = EarlyStopping(patience = args.patience, verbose = True, path = f'{args.output_dir}/ckpt/best_model.pt')


    train_losses = []
    valid_losses = []
    for epoch in range(args.epoch_num):
        print(f'Current Epoch: {epoch+1}')
        model.train()
        train_losses = train_epoch(train_dataloader, optimizer, model, "train")
        model.eval()
        with torch.no_grad():
            valid_losses = train_epoch(val_dataloader, optimizer, model, "valid")
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
                
        print(f'\nEpoch {epoch+1}, train loss: {train_loss}, valid loss: {valid_loss}')
        torch.save(model.state_dict(),f'./{args.output_dir}/ckpt/model_{epoch+1}.pt')
        if (epoch+1 > 15): 
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch+1}')
                break  
        scheduler.step(valid_loss)

def main():
    print(f'Loadding ids from {args.cleaned_path}...')
    start = time.time()
    profile = load_pickle(f'{args.cleaned_path}/profile_embeds.pkl')
    print("Loaded profile ids")
    query = load_pickle(f'{args.cleaned_path}/q_embeds.pkl')
    print("Loaded query ids")
    dialogue = load_pickle(f'{args.cleaned_path}/dialog_embeds.pkl')
    print("Loaded dialogue ids")
    
    end  = time.time()
    print("Total loading time: ", end-start, "s")
    print('Data Statics:')
    print("The length of profiles: ", len(profile))
    print("The length of querys: ", len(query))
    print("The length of dialogues: ", len(dialogue))

    print('Building training dataset and dataloader...')
    train_set = pd.read_csv(f'./dataset/train_cleaned.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    train_dataset = DoctorRecDataset(
        'train', train_set, profile, query, dialogue,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Done')

    print('Building validation dataset and dataloader...')
    valid_set = pd.read_csv(f'./dataset/valid_cleaned.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    val_dataset = DoctorRecDataset(
        'valid', valid_set, profile, query, dialogue,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print('Done')

    model = ourModel().cuda()
    if args.n_gpu > 1 and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model = model.module
    print(next(model.parameters()).device)
    # with LineProfiler(train_model) as prof:
    #     train_model(model, train_dataloader, val_dataloader)
    # prof.display()

    train_model(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()