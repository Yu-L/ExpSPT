import json
import logging
import math
import os
import random

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from dataset.tokenizer import create_vocabulary, Vocabulary, codon_vocab, Tokenizer
from dataset.dataset import Dataset as OptDataset
from models import SPT
from models.metrics import regression_metrics, binary_metrics
from dataset.dataset import ConstantLengthDataset
from torch.utils.data import DataLoader
from models.trainer import trainer

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import numpy as np
from torch import einsum, nn
import torch.nn.functional as F
import optuna
from accelerate import init_empty_weights


"""

downstream task finetune


"""

class Class_SPT(nn.Module):
    """ 
    downstream task finetune network

    默认冻结预训练模型参数
    """
    
    
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        seq_len=128,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        qk_rmsnorm = False,
        lora_r = 8,
        rotary_xpos_scale_base = 512,
        flash_attn = False,
        finetune_scopes = tuple(),
        cross_entropy_ignore_index = 0,
        weight_location = None,
        freeze = True,
        regression = False
    ):
        super().__init__()
        
        
        self.weight_location = weight_location
        
        self.spt = SPT(
            num_tokens= num_tokens, 
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head = dim_head, 
            flash_attn=True
        )
        
        if self.weight_location:
            print(f"loading pretrained model from checkpoint: {self.weight_location}")
            self.spt.load_state_dict(
                torch.load(self.weight_location))    
            # accelerator.load_state(best_metric_checkpoint)

        # freeze pretrain model's parameters
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
                
        self.fc = nn.Linear(dim, 1)
        
        if regression:
            self.fc2 = nn.Linear(seq_len, 1)
        else:
            self.fc1 = nn.Linear(seq_len, 2)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(ff_dropout)
        
        
    def load_pretrain_weights(self):
        """

        """
        if self.weight_location:
            print(f"loading pretrained model from checkpoint: {self.weight_location}")
            self.spt.load_state_dict(
                torch.load(self.weight_location, map_location='auto'))        
        

    def forward(
        self,
        x,
        labels=None,
        return_loss = False,
        disable_lora = False,
        finetune_scope = None,
        extra_embed = None,
        return_only_embedding = False,
        return_logits_with_embedding = False,
        generate = False,
        regression = False
    ):
        
        
        if return_only_embedding:
            return self.fc(self.spt.forward(x, return_only_embedding=True))
    

        flatten_embds = self.fc(self.spt.forward(x, return_only_embedding=True)).squeeze(-1)
        # logits = self.softmax(self.fc1(flatten_embds))
        if regression:
            logits = self.fc2(self.dropout(flatten_embds))
            if generate:
                return logits.squeeze(-1)
            return nn.MSELoss()(logits.squeeze(-1), labels)
        else:
            logits = self.fc1(self.dropout(flatten_embds))
            if generate:
                return logits
            return F.cross_entropy(logits, labels.long())
    




def task_tokenizer(examples, max_len=4096):
        """
        post process for training
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize_to_ids(examples['seq'], seq_type=None)
        pad_encoded= list(tokens) + [0]*(max_len-len(tokens)) if len(tokens) < max_len else list(tokens[:max_len])
        
        examples['X'] = torch.tensor(pad_encoded).long()
        examples['y'] =  examples['value']
        examples = {k:v for k, v in examples.items() if k in ['X', 'y']}
        # print("examples: ", examples)
        return examples


def train_func(model, batch):
    """
    训练
    """
    with torch.cuda.amp.autocast():
        loss = model(batch['X'], labels=batch['y'],  regression=True)
        
    return loss


def eval_func(model, val_loader, accelerator, args):
    """
    验证
    """

    # model.eval()
    trues, outs = [], []
    for step, batch in enumerate(val_loader):

        # if step > 8:
        #     break
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # with accelerator.autocast():
                out = model(batch['X'], labels=batch['y'], generate=True, regression=True)
        outs.append(out)
        trues.append(batch['y'])

    # refs: https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/7_accelerate.html
    outs = torch.stack(outs)
    trues = torch.stack(trues)
    gathered_preds, gathered_trues = accelerator.gather_for_metrics((outs, trues))


    # calculate metrics
    if args.regression:
        metrics_res = regression_metrics(gathered_preds, gathered_trues)
        # all_metrics_res = pd.concat([all_metrics_res, pd.DataFrame(metrics_res, index=[ie])], axis=0)

        metric = metrics_res['R2']
    else:
        y_trues, y_preds, y_probs = gathered_trues, gathered_preds.detach().softmax(dim=1).argmax(axis=1), gathered_preds.detach().softmax(dim=1)[:,1]
        metrics_res = binary_metrics(y_trues, y_preds, y_probs)

        # all_metrics_res = pd.concat([all_metrics_res, pd.DataFrame(metrics_res, index=[ie])], axis=0)

        fpr[ie], tpr[ie], _ = metrics.roc_curve(y_trues, y_probs)
        roc_auc[ie] = metrics.auc(fpr[ie], tpr[ie])

        metric = metrics_res['auc']

    return {
        'metric': metric,
        **metrics_res
        }



def train(args):
    """
    train model for finetune downstream regression tasks
    """

    # regression = True
    # study_number = config['study_number']
    # try_prefix = f'{data_name}_sft{study_number}'
    regression = True

    dropout = 0.5
    device = 'cuda:0' # 'cpu' #'cuda:0' # 'cpu'# 


    CHECKPOINT_PATH = f'checkpoints/train002_10000_state/pytorch_model.bin'
    # summaryWriter = SummaryWriter(f"logs/tensorboard/{args.data_name}")
    logger = args.logger
    # model
    from accelerate import init_empty_weights
    # with init_empty_weights():
    model = Class_SPT(
        num_tokens= 101,
        dim=2048,
        depth=32,
        heads=32,
        dim_head=64,
        flash_attn=True,
        seq_len=4096,
        # weight_location=args.init_from_checkpoint,
        ff_dropout=dropout,
        regression=regression
        )

    print('args.accelerator.device: ', args.accelerator.device)
    # model.spt.load_state_dict(
    #     torch.load(args.init_from_checkpoint), map_location=args.accelerator.device
    #     )
    if args.init_from_checkpoint:
        print(f"Loading pretrain checkpoint from {args.init_from_checkpoint}")
        model.load_state_dict(torch.load(args.init_from_checkpoint, map_location=args.accelerator.device))
    # tokenizer
    tokenizer = Tokenizer()

    # dataset
    dataset = load_from_disk(args.dataset_name)

    train_dataset = dataset['train'].map(task_tokenizer).with_format("torch")    
    valid_dataset = dataset['test'].map(task_tokenizer).with_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.per_device_train_batch_size, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


    # train
    trainer(model, train_loader, val_loader, train_func, eval_func, args)



def parallel_train(trial):
    """
    gridsearch for hyper-parameter
    """
        
    Config = {
        # "gpus": "0,1,2,3,4,5,6,7", # GPUs to use. "0,1" means use GPU 0 and 1
        "batch_size": trial.suggest_categorical('batch_size',[24, 2*24, 4*24]), # 
        "steps": trial.suggest_categorical('steps', [100, 1000]), # [100, 1000, 10000, 100000]), # number of parameter updates
        "lr": 10**trial.suggest_int('lr', -7, -4), # trial.suggest_float('lr', 1e-3, 1e-1, log=True),  # 0.001, # learning rate
        "weight_decay": 10**trial.suggest_int('weight_decay', -8, -3), 
        "dropout": trial.suggest_categorical('dropout',[0.1, 0.3, 0.5, 0.6]), 
        "grad_norm": trial.suggest_categorical('grad_norm',[0.1, 0.5, 1, 2]),
        # "grad_accum": 4, # accumulate gradients for better performance
        "max_len": trial.suggest_categorical('max_len',[128, 256, 512]), 
        "study_number": trial.number,
        
        # "freeze": trial.suggest_categorical('freeze',[True, False]),
        # "sigmoid": trial.suggest_categorical('sigmoid',[True, False])
        
    }
    # os.environ["CUDA_VISIBLE_DEVICES"] = Config["gpus"]
    # world_size = len(Config["gpus"].split(","))
    

    
    return train(
        Config
    )
    



def main():
    """
    train main function for parallel parameter optimaization

    """
    
    study_name = 'LNPreward-sft-direct-partial' # 'LNPreward-finetune'
    url = 'sqlite:////data/user/user_wan/user_project/yul/ChemSPT/example.db'
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

    study.optimize(parallel_train, n_trials=100)

    # # 可视化
    params = ['batch_size','steps','lr','max_len', 'dropout', 'grad_norm']
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