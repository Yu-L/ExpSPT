import os
import gzip
import random
import tqdm
import numpy as np
import optuna

import math
from pathlib import Path
import copy
from functools import partial
from collections import deque, namedtuple
from random import randrange
from copy import deepcopy

from beartype import beartype
from beartype.typing import List, Optional, Callable, Deque


from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange


import torch
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from transformers import default_data_collator, get_linear_schedule_with_warmup


from models import SPT
from models.reward import RewardModel
from models.optimizer import get_optimizer
from models.utils import masked_mean, eval_decorator
# from dataset.tokenizer import create_vocabulary, SMILESTokenizer, Vocabulary
from dataset.dataset import Dataset as OptDataset
from dataset.dataset import ConstantLengthDataset

from datasets import load_dataset, load_from_disk
import itertools
from copy import deepcopy

from datetime import datetime
import pytz


from accelerate import Accelerator

shanghai = pytz.timezone("Asia/Shanghai") 
currentTime = datetime.now(shanghai).strftime("%H:%M:%S")





def train(config):
    """
    train model for DPO
    """



    SAVE_EVERY = int(1e3)
    VALIDATE_EVERY = 1
    PRIME_LENGTH = 12
    GENERATE_EVERY = 50 #500


    NUM_EPOCHS= config['steps'] # 300 # 10000
    SEQ_LEN = 4096 #config['max_len'] # 512
    BATCH_SIZE = config['batch_size'] # 32
    LEARNING_RATE = config['lr'] # 1e-6
    beta = config['beta'] # 0.1
    grad_norm =  config['grad_norm']
    # 超参数
    buffer_size = int(1e5)

    devices = ['cuda:1', 'cuda:2']
    device1 = devices[0] # 'cuda:1' # 'cpu' #
    device2 = devices[1] # 'cuda:2'

    data_name = 'paired_utr3_preference_rank1.5_roughly_diff' # 'paired_mRNA_preference_small' # 'paired_mRNA_preference'
    study_number = config['study_number']
    try_prefix = f'{data_name}_dpo06_T{study_number}'

    summaryWriter = SummaryWriter("logs/tensorboard/")

    # model
    model = SPT(
        num_tokens= 101,
        dim=2048,
        depth=32,
        heads=32,
        dim_head=64,
        flash_attn=True
    ).to(device1)

    # pretrained model ckpt
    PRETRAIN_CHECKPOINT_PATH = 'checkpoints/finetune001_t1/pytorch_model.bin'
    model.load_state_dict(
                    torch.load(PRETRAIN_CHECKPOINT_PATH, map_location=device1)["model"], strict=True)

    # 加载模型
    policy_model = model
    reference_model = deepcopy(policy_model).to(device2)

    REAL_BATCH_SIZE = 2
    GRADIENT_ACCUMULATE_EVERY = BATCH_SIZE // 2 if (BATCH_SIZE // 2) else 1
    print('GRADIENT_ACCUMULATE_EVERY: ', GRADIENT_ACCUMULATE_EVERY)
    train_test_dataset = load_from_disk(os.path.join('data/processed/hf_datasets', data_name))
    train_loader = DataLoader(train_test_dataset['train'].with_format("torch"), batch_size=REAL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(train_test_dataset['test'].with_format("torch"), batch_size=REAL_BATCH_SIZE, shuffle=True)


    # optimizer
    NUM_BATCHES = NUM_EPOCHS*len(train_loader)
    optim = Lion(model.palm_parameters(), lr = LEARNING_RATE)

    scheduler = get_linear_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=0,
                num_training_steps=NUM_BATCHES+1000, 
                )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optim, 
    #     T_max=NUM_BATCHES, 
    #     eta_min=1e-7,
    #     last_epoch=-1, 
    #     verbose='deprecated')

    policy_model.train()
    best_R2 = -100
    t_loss = []
    for ie in tqdm.tqdm(range(NUM_EPOCHS), mininterval=10.0, desc='train samples...'):
        
        running_loss = 0.0
        for i in range(len(train_loader)):

            # 梯度累积
            b_loss = 0
            for _ in range(GRADIENT_ACCUMULATE_EVERY):
                data = next(iter(train_loader))
                input_ids = data['input_ids'].view(-1, SEQ_LEN).to(device1)
                loss_mask = data['loss_mask'].view(-1, SEQ_LEN-1).to(device1)
                labels = data['labels'].view(-1, SEQ_LEN-1).to(device1)

                # 计算 policy model的log prob
                with torch.cuda.amp.autocast():
                    logits, embeds = policy_model(input_ids, return_logits_with_embedding=True)
                    logits = logits[:, :-1, :]
                    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                    all_logps = (per_token_logps * loss_mask).sum(-1)
                    # good response, bad response 间隔交错
                    policy_good_logps, policy_bad_logps =  all_logps[::2], all_logps[1::2]

                # 计算 reference model的log prob
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = reference_model(input_ids.to(device2), return_logits_with_embedding=True)[0][:, :-1, :]
                        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.to(device2).unsqueeze(2)).squeeze(2)
                        all_logps = (per_token_logps * loss_mask.to(device2)).sum(-1)
                    # good response, bad response 间隔交错
                    reference_good_logps, reference_bad_logps = all_logps[::2], all_logps[1::2]

                # 计算loss，会自动进行广播
                logits = (policy_good_logps - reference_good_logps.to(device1)) - (policy_bad_logps - reference_bad_logps.to(device1))
                loss = -F.logsigmoid(beta * logits).mean() # 求期望
            
                accumulated_loss = loss / GRADIENT_ACCUMULATE_EVERY
                # t_loss.append(loss.item())
                loss.backward(accumulated_loss)
                b_loss += accumulated_loss # batch loss
            
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_norm)
            optim.step()
            scheduler.step()
            optim.zero_grad()
            running_loss += b_loss.detach().item()*REAL_BATCH_SIZE
            step_loss = b_loss.detach().item()
            completed_steps = ie*len(train_loader) + i

            summaryWriter.add_scalars(try_prefix + '_training_batch_loss', {
                    'loss': b_loss.detach().item(), }, completed_steps )
            
            print(f"{datetime.now(shanghai)} -- batch: {completed_steps}, lr:{scheduler.get_lr()[0]} training loss: {step_loss}")
        # scheduler.step()
        epoch_loss = running_loss / (len(train_loader) * REAL_BATCH_SIZE)
        summaryWriter.add_scalars(try_prefix + '_training_loss', {
                            'loss': epoch_loss, }, ie )


        # validation
        if ie % VALIDATE_EVERY == 0:
    
            metrics = {
                'reward_accuracies': [],
            }

            for i in range(len(val_loader)):
                data = next(iter(val_loader))
                input_ids = data['input_ids'].view(-1, SEQ_LEN).to(device1)
                loss_mask = data['loss_mask'].view(-1, SEQ_LEN-1).to(device1)
                labels = data['labels'].view(-1, SEQ_LEN-1).to(device1)

                # 计算 policy model的log prob
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits, embeds = policy_model(input_ids, return_logits_with_embedding=True)
                        logits = logits[:, :-1, :]
                        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                        all_logps = (per_token_logps * loss_mask).sum(-1)
                        # good response, bad response 间隔交错
                        policy_good_logps, policy_bad_logps =  all_logps[::2], all_logps[1::2]

                # 计算 reference model的log prob
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = reference_model(input_ids.to(device2), return_logits_with_embedding=True)[0][:, :-1, :]
                        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.to(device2).unsqueeze(2)).squeeze(2)
                        all_logps = (per_token_logps * loss_mask.to(device2)).sum(-1)
                        # good response, bad response 间隔交错
                        reference_good_logps, reference_bad_logps = all_logps[::2], all_logps[1::2]

                logits = (policy_good_logps - reference_good_logps.to(device1)) - (policy_bad_logps - reference_bad_logps.to(device1))

                reward_accuracies = (logits > 0).float()

                metrics['reward_accuracies'].extend(reward_accuracies.cpu().numpy().tolist())

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

                metric =  mean_eval_metrics['reward_accuracies']
                summaryWriter.add_scalars(try_prefix + '_validation_metrics', {
                        'accuracy':  mean_eval_metrics['reward_accuracies']
                    }, ie )

        # save checkpoint
        if best_R2 < metric:
            best_R2 = metric
            FINETUEN_CHECKPOINT_PATH = f'checkpoints/{data_name}/{try_prefix}_finetune_fc2_epoch{NUM_EPOCHS}_{ie}.state_dict.pth'
            # FINETUEN_CHECKPOINT_PATH = f'checkpoints/{data_name}/{try_prefix}_finetune_fc2_epoch{NUM_EPOCHS}.state_dict.pth'
            print(f"save model on metrics {metric}")
            if not os.path.exists(os.path.dirname(FINETUEN_CHECKPOINT_PATH)):
                os.makedirs(os.path.dirname(FINETUEN_CHECKPOINT_PATH))
            torch.save(policy_model.state_dict(), FINETUEN_CHECKPOINT_PATH)

    print("best R2: ", best_R2)

    return best_R2

def parallel_train(trial):
    """
    gridsearch for hyper-parameter
    """
        
    # Config = {
    #     # "gpus": "0,1,2,3,4,5,6,7", # GPUs to use. "0,1" means use GPU 0 and 1
    #     "batch_size": trial.suggest_categorical('batch_size', [4, 8, 16]), 
    #     "steps": trial.suggest_categorical('steps', [1, 3, 5, 10]), # [100, 1000, 10000, 100000]), # number of parameter updates
    #     "lr": 10**trial.suggest_int('lr', -7, -5), # trial.suggest_float('lr', 1e-3, 1e-1, log=True),  # 0.001, # learning rate
    #     # "weight_decay": 10**trial.suggest_int('weight_decay', -8, -3), 
    #     "beta": trial.suggest_categorical('beta',[0.1, 0.3, 0.5, 0.6]), 
    #     "grad_norm": trial.suggest_categorical('grad_norm',[0.1, 0.5, 1, 2]),
    #     # "grad_accum": 4, # accumulate gradients for better performance
    #     # "max_len": trial.suggest_categorical('max_len',[256, 512]), 
    #     "study_number": trial.number,
        
    #     # "freeze": trial.suggest_categorical('freeze',[True, False]),
    #     # "sigmoid": trial.suggest_categorical('sigmoid',[True, False])
        
    # }

    Config = {
        # "gpus": "0,1,2,3,4,5,6,7", # GPUs to use. "0,1" means use GPU 0 and 1
        "batch_size": trial.suggest_categorical('batch_size', [32, 64]), 
        "steps": trial.suggest_categorical('steps', [3, 10]), # [100, 1000, 10000, 100000]), # number of parameter updates
        "lr": 10**trial.suggest_int('lr', -7, -6), # trial.suggest_float('lr', 1e-3, 1e-1, log=True),  # 0.001, # learning rate
        # "weight_decay": 10**trial.suggest_int('weight_decay', -8, -3), 
        "beta": trial.suggest_categorical('beta',[0.5, 0.6]), 
        "grad_norm": trial.suggest_categorical('grad_norm',[ 0.5, 1]),
        "study_number": trial.number,
        
        
    }

    
    return train(
        Config
    )




def main():
    """
    train main function for parallel parameter optimaization

    """
    
    by_handle = True
    if not by_handle:
        study_name = 'mRNAreward-DPO07'
        url = 'sqlite:////data/user/user_wan/user_project/yul/optuna.db'
        # url = 'sqlite:////maindata/data/user/user_wan/yul/bioSPT2/logs/example.db'
        try:
            storage = optuna.storages.RDBStorage(url=url, engine_kwargs={"connect_args": {"timeout": 100}})
            study = optuna.load_study(study_name=study_name, storage=storage) # 'distributed-umpclass-finetune'
        
        except:
            study = optuna.create_study(study_name = study_name, 
                                        direction = 'maximize', 
                                        sampler = optuna.samplers.TPESampler(),
                                        storage = url,
                                        load_if_exists=True
                                        )

        study.optimize(parallel_train, n_trials=40)

        # # 可视化
        params = ['batch_size','steps','lr', 'beta', 'grad_norm']
        optuna.visualization.plot_contour(study)
        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_param_importances(study)
        optuna.visualization.plot_slice(study, params=params)
        optuna.visualization.plot_parallel_coordinate(study,params=params)

        
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        print(df)
        
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("\t{}:{}".format(key, value))

    else:
        Config = {
            # "gpus": "0,1,2,3,4,5,6,7", # GPUs to use. "0,1" means use GPU 0 and 1
            "batch_size": 2,
            "steps": 2,
            "lr": 7e-7,
            # "weight_decay": 10**trial.suggest_int('weight_decay', -8, -3), 
            "beta": 0.3,
            "grad_norm": 0.5,
            # "grad_accum": 4, # accumulate gradients for better performance
            # "max_len": trial.suggest_categorical('max_len',[256, 512]), 
            "study_number": 3,
            
            # "freeze": trial.suggest_categorical('freeze',[True, False]),
            # "sigmoid": trial.suggest_categorical('sigmoid',[True, False])
            
        }

        train(Config)