
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
from torch.utils.data import DataLoader


from dataset.tokenizer import create_vocabulary, Vocabulary, codon_vocab
from dataset.dataset import Dataset as OptDataset
from dataset.dataset import ConstantLengthDataset

from dataset.dataset import CrfDataset
from models.trainer import trainer
from models.CRF import SPT_CRF
from models.metrics import calc_similarity
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import json
import numpy as np
from itertools import chain
import warnings
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', BiopythonDeprecationWarning)

from Bio import SeqIO
import Bio.SeqUtils.CodonUsage
from Bio import pairwise2 as pw2
from Bio.Seq import Seq



codon_vocab_reverse = {v:k for k, v in codon_vocab.items()}


def similarity(true, pred):
    """
    计算 序列一致性
    """
    simi_scores = [calc_similarity(true_seq, pred_seq) for true_seq, pred_seq in zip(true, pred) ]
    
    return simi_scores


def trans_label(seq1, seq2):
    """
    check if aa is changed
    """

    try:
        if Seq(seq1).translate()==Seq(seq2).translate():
            return 1
        else:
            return -1
    except:
        return -1e6

def aa_same(true, pred):

    return [trans_label(seq1, seq2) for seq1, seq2 in zip(true, pred) ]



def train_func(model, batch):
    """
    训练
    """
    input_ids, tag_tensor, tag_mask, c_masks = batch
    with torch.cuda.amp.autocast():
        forward_score, gold_score, best_score, best_paths = model(
                        input_ids, 
                        tag_tensor = tag_tensor,
                        tag_mask = tag_mask,
                        c_masks = c_masks
        )
        loss = (forward_score - gold_score).mean()
    return loss


def eval_func(model, val_loader, accelerator, args):
    """
    验证
    """

    model.eval()
    losses = []
    generated_seqs, tgt_seqs, aa_same_socres, simi_scores = [], [], [], []
    for step, batch in enumerate(val_loader):

        input_ids, tag_tensor, tag_mask, c_masks = batch
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _,_,_, translated_tokens = model(input_ids, 
                                        tag_mask = tag_mask,
                                            c_masks=c_masks
                                        )
        generated_seq = [ ''.join([codon_vocab_reverse[a] for a in _tokens if codon_vocab_reverse[a] !="<pad>"]) for _tokens in translated_tokens]
        tgt_seq = [ ''.join([codon_vocab_reverse[a] for a in _tokens if codon_vocab_reverse[a] !="<pad>" ]) for _tokens in tag_tensor.cpu().numpy().tolist()]
      
        simi_scores.extend(similarity(generated_seq, tgt_seq))
        generated_seqs.extend(generated_seq)
        tgt_seqs.extend(tgt_seq)
        aa_same_socres.extend(aa_same(generated_seq, tgt_seq))


    gathered_simi_scores = accelerator.gather_for_metrics(simi_scores)
    gathered_generated_seqs = accelerator.gather_for_metrics(generated_seqs)
    gathered_tgt_seqs = accelerator.gather_for_metrics(tgt_seqs)
    gathered_aa_same_socres = accelerator.gather_for_metrics(aa_same_socres)

    if accelerator.is_main_process:
        # save
        save_f = f'{args.output_dir}/metrics_df_{accelerator.process_index}.json'
        with open(save_f, 'w') as f:
            json.dump({
                "simi_scores": gathered_simi_scores,
                "generated_seqs": gathered_generated_seqs,
                "tgt_seqs": gathered_tgt_seqs,
                "aa_same_socres": gathered_aa_same_socres

            }, f)

        accelerator.print(f"Save validation results to {save_f}")

    return {
            'metric': np.mean(gathered_simi_scores),
            'simi_scores': np.mean(gathered_simi_scores), 
            'aa_same_socres': np.mean(gathered_aa_same_socres),
            }




def train(args):



    # model 
    model = SPT_CRF(
        tagset_size=68,  # condon_vocab size
        hidden_dim = 2048,
        freeze = True
        # model_weights = 'checkpoints/paired_mRNA_preference_rank1.5_roughly_diff_dpo06_T7/step_1800_checkpoint/pytorch_model.bin' # 'checkpoints/finetune001_t1/pytorch_model.bin'
    )

    # args.init_from_checkpoint = '/maindata/data/user/user_wan/yul/bioSPT2/checkpoints/CRF_finetune/model.state_dict.pth'
    # if args.init_from_checkpoint:
    #     print(f"Loading pretrain checkpoint from {args.init_from_checkpoint}")
    #     model.load_state_dict(torch.load(args.init_from_checkpoint))

    
    # data prepare
    dataset = load_from_disk(args.dataset_name)
    dataset = dataset.filter(lambda example: len(example["cds"]) % 3 == 0)


    train_dataset = CrfDataset(dataset['train'], max_seq_length=args.max_seq_len)
    val_dataset  = CrfDataset(dataset['test'], max_seq_length=args.max_seq_len)


    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True,
                                  drop_last=True, num_workers=args.preprocessing_num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, shuffle=True,  
                                drop_last=True, num_workers=args.preprocessing_num_workers, pin_memory=False)



    # train
    trainer(model, train_loader, val_loader, train_func, eval_func, args)




if __name__ == "__main__":
    train()