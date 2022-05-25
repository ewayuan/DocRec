import argparse
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import string
import sys
import time
import json
import math
import copy
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from utils.util import *
from functions import *
from utils.dataset import DoctorRecDataset

from gensim.models import LdaModel
from pprint import pprint
from contextlib import redirect_stdout
from nltk.corpus import stopwords
from gensim.test.utils import datapath

import logging

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")



def main(config, progress):
    # save config
    with open("./log/configs.json", "a") as f:
        json.dump(config, f)
        f.write("\n")
    cprint("*"*80)
    cprint("Experiment progress: {0:.2f}%".format(progress*100))
    cprint("*"*80)
    metrics = {}

    # data hyper-params
    train_path = config["train_path"]
    valid_path = config["valid_path"]
    test_path = config["test_path"]
    dataset = train_path.split("/")[3]
    test_mode = bool(config["test_mode"])
    load_model_path = config["load_model_path"]
    save_model_path = config["save_model_path"]
    num_candidates = config["num_candidates"]
    num_personas = config["num_personas"]
    persona_path = config["persona_path"]
    max_sent_len = config["max_sent_len"]
    max_seq_len = config["max_seq_len"]
    PEC_ratio = config["PEC_ratio"]
    train_ratio = config["train_ratio"]
    if PEC_ratio != 0 and train_ratio != 1:
        raise ValueError("PEC_ratio or train_ratio not qualified!")

    # model hyper-params
    config_id = config["config_id"]
    bert_model = config["model"]
    shared = bool(config["shared"])
    apply_interaction = bool(config["apply_interaction"])
    matching_method = config["matching_method"]
    aggregation_method = config["aggregation_method"]
    output_hidden_states = False

    # training hyper-params
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    warmup_steps = config["warmup_steps"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    lr = config["lr"]
    weight_decay = 0
    seed = config["seed"]
    device = torch.device(config["device"])
    fp16 = bool(config["fp16"])
    fp16_opt_level = config["fp16_opt_level"]

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if test_mode and load_model_path == "":
        raise ValueError("Must specify test model path when in test mode!")

    # load data
    print(f'Loadding ids from {args.cleaned_path}...')
    start = time.time()
    profile_ids = load_pickle(f'{args.cleaned_path}/profile_ids_mini.pkl')
    print("Loaded profile ids")
    query_ids = load_pickle(f'{args.cleaned_path}/q_ids_mini.pkl')
    print("Loaded query ids")
    dialogue_ids = load_pickle(f'{args.cleaned_path}/dialog_ids_mini.pkl')
    print("Loaded dialogue ids")
    end  = time.time()
    print("Total loading time: ", end-start, "s")
    print('Data Statics:')
    print("The length of profiles: ", len(profile_ids))
    print("The length of querys: ", len(query_ids))
    print("The length of dialogues: ", len(dialogue_ids))

    print('Building training dataset and dataloader...')
    train_set = pd.read_csv(f'./dataset/train_mini.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})

    if not test_mode:
        train_dataset = DoctorRecDataset(
        'train', train_set, profile_ids, query_ids, dialogue_ids,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    del train_set, train_dataset
    print('Done')
        t_total = len(train_dataloader) // gradient_accumulation_steps * epochs

    valid_dataset = TensorDataset(all_context_ids_valid, all_context_attention_mask_valid, all_context_token_type_ids_valid, \
                                  all_response_ids_valid, all_response_attention_mask_valid, all_response_token_type_ids_valid, \
                                  all_persona_ids_valid, all_persona_attention_mask_valid, all_persona_token_type_ids_valid,\
                                  context_topic_distribution_valid, response_topic_distribution_valid, persona_topic_distribution_valid)
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=num_candidates)


    # Create model
    cprint("Building dmodel...")

    model = ourModel()
    cprint(model)
    cprint("number of parameters: ", count_parameters(model))

    if shared: # responses, persona, context share same bert, else, they can have their own bert models.
        cprint("number of encoders: 1")
        models = [model]
    else:
        if num_personas == 0:
            cprint("number of encoders: 2")
            # models = [model, copy.deepcopy(model)]
            models = [model, pickle.loads(pickle.dumps(model))]
        else:
            cprint("number of encoders: 3")
            # models = [model, copy.deepcopy(model), copy.deepcopy(model)]
            models = [model, pickle.loads(pickle.dumps(model)), pickle.loads(pickle.dumps(model))]

    if test_mode:
        cprint("Loading weights from ", load_model_path)
        checkpoint = torch.load(load_model_path)
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)
        # model.module.load_state_dict(torch.load(load_model_path))
        models = [model]
        cprint("loaded weight successuly")

    for i, model in enumerate(models):
        cprint("model {0} number of parameters: ".format(i), count_parameters(model))
        model.to(device)

    # optimization
    amp = None
    if fp16:
        from apex import amp

    no_decay = ["bias", "LayerNorm.weight"]
    optimizers = []
    schedulers = []
    cprint("debugger")
    for i, model in enumerate(models):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, weight_decay=0)

        if fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
            if not test_mode:
                models[i] = nn.DataParallel(model, device_ids=[0])
            else:
                models[i] = nn.DataParallel(model, device_ids=[0])
            cprint("model into DataParallel")
        optimizers.append(optimizer)

        if not test_mode:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            schedulers.append(scheduler)

    if test_mode:
        cprint("model.eval()")
        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = model.module.evaluate_epoch(valid_iterator, models, \
            num_personas, gradient_accumulation_steps, device, dataset, 0, apply_interaction, matching_method, aggregation_method)
        cprint("test loss: {0:.4f}, test acc: {1:.4f}, test recall: {2}, test MRR: {3:.4f}"
            .format(valid_loss, valid_acc, valid_recall, valid_MRR))
        sys.exit()

    # training
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_accs = []
    epoch_valid_recalls = []
    epoch_valid_MRRs = []
    cprint("***** Running training *****")
    cprint("Num examples =", len(train_dataset))
    cprint("Num Epochs =", epochs)
    cprint("Total optimization steps =", t_total)
    best_model_statedict = {}
    for epoch in range(epochs):
        cprint("Epoch", epoch+1)
        # training
        for model in models:
            model.train()
        train_iterator = tqdm(train_dataloader, desc="Iteration")

        train_loss, (train_acc, _, _) = model.module.train_epoch(train_iterator, models, num_personas, optimizers, \
            schedulers, gradient_accumulation_steps, device, fp16, amp, apply_interaction, matching_method, aggregation_method)
        epoch_train_losses.append(train_loss)

        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = model.module.evaluate_epoch(valid_iterator, models, \
            num_personas, gradient_accumulation_steps, device, dataset, epoch, apply_interaction, matching_method, aggregation_method)

        cprint("Config id: {7}, Epoch {0}: train loss: {1:.4f}, valid loss: {2:.4f}, train_acc: {3:.4f}, valid acc: {4:.4f}, valid recall: {5}, valid_MRR: {6:.4f}"
            .format(epoch+1, train_loss, valid_loss, train_acc, valid_acc, valid_recall, valid_MRR, config_id))

        epoch_valid_losses.append(valid_loss)
        epoch_valid_accs.append(valid_acc)
        epoch_valid_recalls.append(valid_recall)
        epoch_valid_MRRs.append(valid_MRR)

        if save_model_path != "":
            if epoch == 0:
                for k, v in models[0].state_dict().items():
                    best_model_statedict[k] = v.cpu()
            else:
                if epoch_valid_recalls[-1][0] == max([recall1 for recall1, _, _ in epoch_valid_recalls]):
                    for k, v in models[0].state_dict().items():
                        best_model_statedict[k] = v.cpu()


    config.pop("seed")
    config.pop("config_id")
    metrics["config"] = config
    metrics["score"] = max(epoch_valid_accs)
    metrics["epoch"] = np.argmax(epoch_valid_accs).item()
    metrics["recall"] = epoch_valid_recalls
    metrics["MRR"] = epoch_valid_MRRs

    if save_model_path:
        cprint("Saving model to ", save_model_path)
        # cprint("best_model_statedict: ", best_model_statedict)
        torch.save(best_model_statedict, save_model_path)

    return metrics


def clean_config(configs):
    cleaned_configs = []
    for config in configs:
        if config not in cleaned_configs:
            cleaned_configs.append(config)
    return cleaned_configs


def merge_metrics(metrics):
    avg_metrics = {"score" : 0}
    num_metrics = len(metrics)
    for metric in metrics:
        for k in metric:
            if k != "config":
                avg_metrics[k] += np.array(metric[k])

    for k, v in avg_metrics.items():
        avg_metrics[k] = (v/num_metrics).tolist()

    return avg_metrics


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Model for Transformer-based Dialogue Generation with Controlled Emotion")
    parser.add_argument('--config', help='Config to read details', required=True)
    parser.add_argument('--note', help='Experiment note', default="")
    args = parser.parse_args()
    cprint("Experiment note: ", args.note)
    with open(args.config) as configfile:
        config = json.load(configfile) # config is now a python dict

    # pass experiment config to main
    parameters_to_search = OrderedDict() # keep keys in order
    other_parameters = {}
    keys_to_omit = ["kernel_sizes"] # keys that allow a list of values
    for k, v in config.items():
        # if value is a list provided that key is not device, or kernel_sizes is a nested list
        if isinstance(v, list) and k not in keys_to_omit:
            parameters_to_search[k] = v
        elif k in keys_to_omit and isinstance(config[k], list) and isinstance(config[k][0], list):
            parameters_to_search[k] = v
        else:
            other_parameters[k] = v

    if len(parameters_to_search) == 0:
        config_id = time.perf_counter()
        config["config_id"] = config_id
        cprint(config)
        output = main(config, progress=1)
        cprint("-"*80)
        cprint("config: ", output["config"])
        cprint("epoch: ", output["epoch"])
        cprint("score: ", output["score"])
        cprint("recall: ", output["recall"])
        cprint("MRR: ", output["MRR"])
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]

            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}
            all_configs.append(merged_config)

        #   cprint all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            cprint(config)
            cprint("\n")

        # multiprocessing
        num_configs = len(all_configs)
        # mp.set_start_method('spawn')
        pool = mp.Pool(processes=config["processes"])
        results = [pool.apply_async(main, args=(x,i/num_configs)) for i,x in enumerate(all_configs)]
        outputs = [p.get() for p in results]

        # if run multiple models using different seed and get the averaged result
        if "seed" in parameters_to_search:
            all_metrics = []
            all_cleaned_configs = clean_config([output["config"] for output in outputs])
            for config in all_cleaned_configs:
                metrics_per_config = []
                for output in outputs:
                    if output["config"] == config:
                        metrics_per_config.append(output)
                avg_metrics = merge_metrics(metrics_per_config)
                all_metrics.append((config, avg_metrics))
            # log metrics
            cprint("Average evaluation result across different seeds: ")
            for config, metric in all_metrics:
                cprint("-"*80)
                cprint(config)
                cprint(metric)

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for config, metric in all_metrics:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(config) + "\n")
                    f.write(json.dumps(metric) + "\n")

        else:
            for output in outputs:
                cprint("-"*80)
                cprint(output["config"])
                cprint(output["score"])
                cprint(output["recall"])
                cprint(output["MRR"])
                cprint("Best result at epoch {0}: ".format(output["epoch"]))
                cprint(output["recall"][output["epoch"]], output["MRR"][output["epoch"]])

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for output in outputs:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(output) + "\n")