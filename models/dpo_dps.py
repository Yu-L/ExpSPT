import os
import sys
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
import transformers
from transformers import default_data_collator, get_linear_schedule_with_warmup
from transformers import SchedulerType,  get_scheduler

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
from pytz import timezone
import logging
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from accelerate.logging import get_logger

def timetz(*args):
    return datetime.now(tz).timetuple()

tz = timezone('Asia/Shanghai') # UTC, Asia/Shanghai, Europe/Berlin
logger = get_logger(__name__)

class Property(object):
    
    def __init__(self, config):
        
        for k,v in config.items():
            setattr(self, k, v)
            
        

def evaluate(args, policy_model, reference_model, val_loader, accelerator, max_seq_len=4096):

    metrics = {
        'reward_accuracies': [],
    }
    import json

    inputs = []
    for i in range(len(val_loader)):
        data = next(iter(val_loader))
        input_ids = data['input_ids'].view(-1, max_seq_len)
        loss_mask = data['loss_mask'].view(-1, max_seq_len-1)
        labels = data['labels'].view(-1, max_seq_len-1)

        print("input_ids: ", input_ids.cpu().numpy())
        # metrics_df.loc[i, 'input_ids'] = input_ids.detach().cpu().numpy()[0]
        inputs.append(input_ids.detach().cpu().numpy().tolist())
        # metrics_df.loc[i, 'loss_mask'] = loss_mask
        # metrics_df.loc[i, 'labels'] = labels
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
                logits = reference_model(input_ids, return_logits_with_embedding=True)[0][:, :-1, :]
                per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                all_logps = (per_token_logps * loss_mask).sum(-1)
                # good response, bad response 间隔交错
                reference_good_logps, reference_bad_logps = all_logps[::2], all_logps[1::2]

        logits = (policy_good_logps - reference_good_logps) - (policy_bad_logps - reference_bad_logps)

        reward_accuracies = (logits > 0).float()
        metrics['reward_accuracies'].extend(reward_accuracies.cpu().numpy().tolist())
        mean_eval_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        metric =  mean_eval_metrics['reward_accuracies']
        # accelerator.print(f" process: {dist.get_rank()} -- validation step {i} metric {metric}")
        print(f" process: {dist.get_rank()} -- validation step {i} metric {metric}")

        with open(f'metrics_df_{dist.get_rank()}.json', 'w') as f:
            json.dump({
                'inputs': inputs,
                'reward_accuracies':metrics['reward_accuracies'],
            }, f)
        
    return metric


def train(config=None, args=None):
    """
    train model for DPO
    """


    VALIDATE_EVERY = 1
    PRIME_LENGTH = 12
    GENERATE_EVERY = 50 #500



    # 超参数
    buffer_size = int(1e5)
    start_step = 0



    summaryWriter = SummaryWriter("logs/tensorboard/")

    if config:
        args = Property(config)
        data_name = 'paired_mRNA_preference_rank1.5_roughly_diff' #'paired_mRNA_preference_small' # 'paired_mRNA_preference'
        study_number = config['study_number']
        try_prefix = f'{data_name}_dpo06_T{study_number}'

        args.model_name = f'{data_name}_dpo06_T{study_number}'

        args.beta = config['beta'] # 0.1
        grad_norm =  config['grad_norm']
        args.weight_decay = 0.001
        args.num_train_epochs= config['steps'] # 300 # 10000
        max_seq_len = 4096 #config['max_len'] # 512
        BATCH_SIZE = config['batch_size'] # 32
        LEARNING_RATE = config['lr'] # 1e-6
        args.learning_rate = LEARNING_RATE
        args.num_warmup_steps = 0
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = max(BATCH_SIZE // (args.per_device_train_batch_size * accelerator.num_processes),  1)
        args.dataset_name = os.path.join('data/processed/hf_datasets', data_name)
        
        args.output_dir = f"checkpoints/{args.model_name}"
        # accelerator
        accelerator = Accelerator(
                log_with='tensorboard',
                project_dir=args.output_dir,
            )
            
        # pretrained model ckpt
        args.init_from_checkpoint = 'checkpoints/finetune001_t1/pytorch_model.bin'
        
        # PRETRAIN_CHECKPOINT_PATH2 = 'checkpoints/paired_mRNA_preference_small_dpo06_T3/paired_mRNA_preference_small_dpo06_T3_epoch2_0_pytorch_model.bin'
        # RETAIN_CHECKPOINT_PATH = '/maindata/data/user/user_wan/yul/bioSPT2/checkpoints/paired_mRNA_preference_rank1.5_roughly_diff_dpo06_T6/step_1800_checkpoint/pytorch_model_1.bin'
        args.resume_from_checkpoint = '/maindata/data/user/user_wan/yul/bioSPT2/checkpoints/paired_mRNA_preference_rank1.5_roughly_diff_dpo06_T7/step_1800_checkpoint'
    
    accelerator = args.accelerator
    device = accelerator.device
    # log
    log_dir = f"logs/tensorboard/{args.model_name}"
    summaryWriter = SummaryWriter(log_dir)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.Formatter.converter = timetz
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        experiment_config["accelerator"] = None 
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # model
    policy_model = SPT(
        num_tokens= 101,
        dim=2048,
        depth=32,
        heads=32,
        dim_head=64,
        flash_attn=True
    )


    # dataset
    train_test_dataset = load_from_disk(args.dataset_name)
    train_loader = DataLoader(train_test_dataset['train'].with_format("torch"), batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(train_test_dataset['test'].with_format("torch"), batch_size=args.per_device_train_batch_size, shuffle=True)

    # 
    total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs*len(train_loader) if args.num_train_epochs else args.max_train_steps

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in policy_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in policy_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)


    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )
        

    # optimizer = Lion(model.palm_parameters(), lr = LEARNING_RATE)

    # scheduler = get_linear_schedule_with_warmup(
    #             optimizer=optim,
    #             num_warmup_steps=100,
    #             num_training_steps=args.max_train_steps+1000, 
    #             )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optim, 
    #     T_max=args.max_train_steps, 
    #     eta_min=1e-7,
    #     last_epoch=-1, 
    #     verbose='deprecated')

    def param_head(model):

        it = 0
        for _, param in enumerate(model.parameters()):
            it+=1
            
            if it > 2:
                break
        return param


    # accelerator.print('policy_model befoe:', param_head(policy_model))
    # accelerator.print('reference_model befoe:', param_head(reference_model))

    # policy_model.load_state_dict(
    #                 torch.load(args.init_from_checkpoint, map_location=device)["model"], strict=True)

    reference_model = deepcopy(policy_model)
    # reference_model.load_state_dict(
    #                 torch.load(args.init_from_checkpoint, map_location=device)["model"], strict=True)

    
    # accelerator.print('policy_model after:', param_head(policy_model))
    # accelerator.print('reference_model after:', param_head(reference_model))
    

   # prepare for accelerator
    policy_model, reference_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        policy_model, reference_model, optimizer, train_loader, val_loader, scheduler
    )

    # load from state
    if args.resume_from_checkpoint:
        accelerator.print('Load from saved checkpoint :', args.resume_from_checkpoint)
        accelerator.load_state(args.resume_from_checkpoint)

        # start_epoch
        # start_step = 1800
        for i in range(start_step):
            data = next(iter(train_loader))

    
    # log
    if accelerator.is_main_process:
        accelerator.print(f'''***** Running training *****:
            Dataset:                  {args.dataset_name}
            Num examples:             {len(train_loader)*args.per_device_train_batch_size}
            Max_train_steps:          {args.max_train_steps}
            Num Epochs:               {args.num_train_epochs}
            Per_device_train_batch_size:      {args.per_device_train_batch_size}
            Gradient_accumulation_steps: {args.gradient_accumulation_steps}
            Total train batch size (w. parallel, distributed & accumulation): {total_batch_size}
            Learning rate:   {args.learning_rate}
            Weight_decay:    {args.weight_decay}
            Max seq lenghth:         {args.max_seq_len}
            Device Num:          {device}
            GPU Nums:            {accelerator.num_processes}
            Model output dir：    {args.output_dir}
        ''')

    validation_only = False
    if validation_only:
        metric = evaluate(args, policy_model, reference_model, val_loader, accelerator)
        print(f"validation metric: {metric}")
        sys.exit()


    policy_model.train()
    best_R2 = -100
    t_loss = []
    # args.num_train_epochs = args.max_train_steps//len(train_loader)
    for ie in tqdm.tqdm(range(0, args.num_train_epochs), mininterval=10.0, desc='train samples...'):
        
        running_loss = 0.0
        for i in range(start_step, len(train_loader)):
            
            # break
            # 梯度累积
            b_loss = 0
            for _ in range(args.gradient_accumulation_steps):
                data = next(iter(train_loader))
                input_ids = data['input_ids'].view(-1, args.max_seq_len)
                loss_mask = data['loss_mask'].view(-1, args.max_seq_len-1)
                labels = data['labels'].view(-1, args.max_seq_len-1)

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
                        logits = reference_model(input_ids, return_logits_with_embedding=True)[0][:, :-1, :]
                        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                        all_logps = (per_token_logps * loss_mask).sum(-1)
                    # good response, bad response 间隔交错
                    reference_good_logps, reference_bad_logps = all_logps[::2], all_logps[1::2]

                # 计算loss，会自动进行广播
                logits = (policy_good_logps - reference_good_logps) - (policy_bad_logps - reference_bad_logps)
                loss = -F.logsigmoid(args.beta * logits).mean() # 求期望
            
                accumulated_loss = loss / args.gradient_accumulation_steps
                # t_loss.append(loss.item())
                # loss.backward(accumulated_loss)
                accelerator.backward(accumulated_loss)
                b_loss += accumulated_loss
            
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            completed_steps = ie*len(train_loader) + i
            step_loss = accelerator.reduce(b_loss, reduction="mean").item()
  
            running_loss += b_loss.detach().item()*args.per_device_train_batch_size
            # summaryWriter.add_scalars(try_prefix + '_training_batch_loss', {
            #         'loss': b_loss.detach().item(), }, ie*len(train_loader)+i )
            if args.with_tracking:
                accelerator.log(
                    {'step_loss': b_loss.detach().item(), }, 
                    step=ie*len(train_loader)+i
                )
            
            accelerator.print(f" process: {dist.get_rank()} -- batch: {completed_steps}, lr:{scheduler.get_lr()[0]} training loss: {step_loss}")

            if completed_steps % args.checkpointing_steps == 0:
                save_checkpoint = os.path.join(args.output_dir, f"step_{completed_steps}_checkpoint")
                accelerator.save_state(save_checkpoint)
                accelerator.print(f"save checkpoint: {save_checkpoint} at training step {completed_steps}")
       
            
        # scheduler.step()
        epoch_loss = running_loss / (len(train_loader) * args.per_device_train_batch_size)
        # summaryWriter.add_scalars(try_prefix + '_training_loss', {
        #                     'loss': epoch_loss, }, ie )
        if args.with_tracking:
            accelerator.log({'epoch_loss': epoch_loss, }, step=ie )

        # validation
        if ie % VALIDATE_EVERY == 0:
            metrics = {
                'reward_accuracies': [],
            }

            for i in range(len(val_loader)):
                data = next(iter(val_loader))
                input_ids = data['input_ids'].view(-1, args.max_seq_len)
                loss_mask = data['loss_mask'].view(-1, args.max_seq_len-1)
                labels = data['labels'].view(-1, args.max_seq_len-1)

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
                        logits = reference_model(input_ids, return_logits_with_embedding=True)[0][:, :-1, :]
                        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                        all_logps = (per_token_logps * loss_mask).sum(-1)
                        # good response, bad response 间隔交错
                        reference_good_logps, reference_bad_logps = all_logps[::2], all_logps[1::2]

                logits = (policy_good_logps - reference_good_logps) - (policy_bad_logps - reference_bad_logps)

                reward_accuracies = (logits > 0).float()

                metrics['reward_accuracies'].extend(reward_accuracies.cpu().numpy().tolist())

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

                metric =  mean_eval_metrics['reward_accuracies']
                # summaryWriter.add_scalars(try_prefix + '_validation_metrics', {
                #         'accuracy':  mean_eval_metrics['reward_accuracies']
                #     }, ie )
                if args.with_tracking:
                    accelerator.log({'mean_accuracy': mean_eval_metrics['reward_accuracies'], }, step=ie )
                accelerator.print(f" process: {dist.get_rank()} -- validation step {i} metric {metric} on epoch {ie}")

        # save checkpoint
        if best_R2 < metric:
            # break
            best_R2 = metric
            final_checkpoint = os.path.join(args.output_dir, f'{args.model_name}_epoch{args.num_train_epochs}_{ie}_pytorch_model.bin')
            if not os.path.exists(os.path.dirname(final_checkpoint)):
                os.makedirs(os.path.dirname(final_checkpoint))

            if final_checkpoint is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(policy_model)

                # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
                # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
                # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
                # For Zero Stages 1 and 2, models are saved as usual in the output directory.
                # The model name saved is `pytorch_model.bin`
    
                if accelerator.is_main_process:
                    accelerator.save({
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
                        },  final_checkpoint)
                    print(f"Finish training on metric {best_R2} and save best model to {final_checkpoint}")
                
                best_metric_checkpoint = os.path.join(args.output_dir, "best_checkpoint")
                accelerator.save_state(best_metric_checkpoint)
                accelerator.print(f"New best metric: {best_R2} at epoch {ie}")
                accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")


    print("best R2: ", best_R2)

    return best_R2


def parallel_train(trial):
    """
    gridsearch for hyper-parameter
    """
        
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
    
    by_handle = True # False
    if not by_handle:
        study_name = 'mRNAreward-DPO07'
        url = 'sqlite:////data/user/user_wan/user_project/yul/optuna.db'
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
            "batch_size": 32,
            "steps": 2,
            "lr": 7e-7,
            # "weight_decay": 10**trial.suggest_int('weight_decay', -8, -3), 
            "beta": 0.3,
            "grad_norm": 0.5,
            "study_number": 7,  
        }

        train(Config)



if __name__ == "__main__":
    main()