
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
from dataset.dataset import ConstantLengthDataset
from torch.utils.data import DataLoader
from models.trainer import trainer

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import json
import numpy as np


"""
pretrain model
"""
def Customtokenizer(examples, pading=False, 
                    max_len=4096, integrity=False, truncate=False):
    """
    custom tokenizer for smiles

    params:
        pading:
        max_len:
        integrity: 确保输入是完整的一段预设语料结构，并且不足用pading补齐，超出则丢弃本记录
        truncate: 截断
    """

    tokenizered = []
    all_encoded = []
    for example in examples:
        tokens = example
        # print("tokens: ", tokens)
        encoded = vocabulary.encode(tokens)

        if truncate:
            encoded = list(encoded[:max_len])
        else:
            encoded = list(encoded)
            if len(encoded) > max_len:
                continue

        if integrity: # check for intergity
            if len(encoded) + len(all_encoded) <= max_len:
                all_encoded.extend(encoded)
                continue
            else:
                n_encoded = encoded
                encoded = all_encoded
                all_encoded = n_encoded

        if pading:
            encoded= list(encoded)+ [0]*(max_len-len(encoded))

        tokenizered.append(encoded)

    return tokenizered

def Directtokenizer(examples):
    """
    custom tokenizer for smiles
    """
      
    return examples

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data


def train_func(model, batch):
    """
    训练
    """

    with torch.cuda.amp.autocast():
        loss = model(batch, return_loss = True)
        
    return loss



def eval_func(model, val_loader, accelerator, args):
    """
    验证
    """

    model.eval()
    losses = []
    for step, batch in enumerate(val_loader):

        if step > 8:
            break
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss = model(batch, return_loss = True)
        losses.append(loss)

    losses = torch.stack(losses)
    gathered_losses = accelerator.gather_for_metrics(losses)

    try:
        eval_loss = torch.mean(gathered_losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")


    return {
            'metric': -perplexity,
            'perplexity': perplexity,
            'loss': eval_loss.item(), 
            }




def train(args):


    # model 
    model = SPT(
        num_tokens= 256,
        dim=4096, # 4096=6.99B, 2048=1.8B
        depth=32,
        heads=32,
        dim_head=64,
        flash_attn=True
    )

 
    if args.init_from_checkpoint:
        print(f"Loading pretrain checkpoint from {args.init_from_checkpoint}")
        model.load_state_dict(torch.load(args.init_from_checkpoint))

    
    # tokenizer
    tokenizer = Tokenizer()

    # data prepare
    # with accelerator.main_process_first():
    # data_train = load_from_disk(args.dataset_name)['train']
    data_train = load_dataset(path=args.dataset_name, split="train", streaming=True)# num_proc=num_proc)
    data_train.with_format("torch")

    data_val = load_dataset(path=args.dataset_name, split="test", streaming=True)
    data_val.with_format("torch")

    # data loader 
    train_dataset = ConstantLengthDataset(
                Directtokenizer, 
                data_train,
                infinite=True,
                content_field='tokens_ids',
                chars_per_token=1,
                max_seq_length=args.max_seq_len,
                data_size=965696,
                accelerator = args.accelerator,
                num_of_sequences=10
            )

    val_dataset = ConstantLengthDataset(
                Directtokenizer, 
                data_val,
                infinite=False,
                content_field='tokens_ids',
                chars_per_token=1,
                max_seq_length=args.max_seq_len,
                data_size=50827,
                accelerator = args.accelerator,
                num_of_sequences=10
            )

    # shuffle
    shuffle=False
    if shuffle:
        buffer_size = int(1e3)
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
        val_dataset = val_dataset.shuffle(buffer_size=buffer_size)


    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=shuffle,
                                  drop_last=True, num_workers=args.preprocessing_num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, shuffle=shuffle,  
                                drop_last=True, num_workers=args.preprocessing_num_workers, pin_memory=False)

    # train
    trainer(model, train_loader, val_loader, train_func, eval_func, args)




if __name__ == "__main__":
    train()