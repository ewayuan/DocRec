import argparse
import json
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from Modules.MLP import MLP
from utils.dataset import MulAttDataset
from utils.config import init_opts, train_opts, eval_opts, multihead_att_opts

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default="med-bert", type=str)
parser.add_argument('--cleaned_path', default='./cleaned', type=str)

parser.add_argument('--dr_dialog_sample', default=2, type=int)
parser.add_argument('--neg_sample', default=2, type=int)
parser.add_argument('--batch_size', default=2, type=int)
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

print(f'Prediction: randome seed {args.seed}, experiment name: {args.name}, run on gpu {args.gpu}')

def main():
    print(f'{args.name} start!')
    print(f'Loadding embeddings from {args.ids_path}...')
    profile_ids = load_pickle(f'{args.cleaned_path}/profile_ids_mini.pkl')
    print("Loaded profile ids")
    query_ids = load_pickle(f'{args.cleaned_path}/q_ids_mini.pkl')
    print("Loaded query ids")
    dialogue_ids = load_pickle(f'{args.cleaned_path}/dialog_ids_mini.pkl')
    print("Loaded dialogue ids")
    end  = time.time()
    
    print('Building test dataset and dataloader...')
    test_set = pd.read_csv(f'./dataset/test_mini.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    test_dataset = DoctorRecDataset(
        'test', test_set, profile_ids, query_ids, dialogue_ids,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    del test_set, test_dataset, profile_ids, query_ids, dialogue_ids,
    print('Done')
    
    model = ourModel()
    model_path = f'{args.output_dir}/ckpt/{args.eval_model}'
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    print(f'{args.name} start prediction...')
    with open(f'{args.output_dir}/test_{args.eval_model}_score.txt', 'w', encoding='utf-8') as score:
        with torch.no_grad():
                pred_scores = test_process(test_dataloader, model)
                for pred_score in pred_scores:
                    print(pred_score.cpu().detach().numpy().tolist()[0], file = score)

if __name__ == '__main__':
    main()