
import json
import logging
import math
import os
import random
import itertools
from copy import deepcopy
from itertools import chain
from pathlib import Path
import gzip
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn import functional as F
from transformers import default_data_collator, get_linear_schedule_with_warmup

from dataset.tokenizer import create_vocabulary, Vocabulary, codon_vocab, Tokenizer, codon_vocab_reverse
from dataset.dataset import PromptDataset
from models import SPT
from models.CRF import CrfLM, SPT_CRF
from models.utils import _parameter_number, param_head


from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed, gather_object

from datetime import datetime
from pytz import timezone
import time





def timetz(*args):
    return datetime.now(tz).timetuple()

tz = timezone('Asia/Shanghai') # UTC, Asia/Shanghai, Europe/Berlin
logger = get_logger(__name__)


start_tokens = {
    'UTR5': '<UTR5_ST>',
    'UTR3': '<UTR3_ST>',
    'CDS': '<CDS_ST>',
    'mRNA': '<UTR5_ST>'
}

max_new_tokens = 4096

def predict_func(model, tokenizer, prompt, accelerator):

    prt, gen_type, temperature = prompt
    outputs = {
        'prompt': prt,
        'gen_type': gen_type,
        'temperature':temperature
    }

    device = accelerator.device
    if gen_type == 'mRNA':
      
        input_ids, tag_mask, c_masks = PromptDataset(prt, max_seq_length=4096).tensor()
        gen_type == 'UTR5'
 
        with torch.no_grad():
            # sample for cds
            cds_scores, output_tokenized = model.generate(
                                    input_ids=input_ids.to(device), 
                                    tag_mask = tag_mask.to(device),
                                    c_masks=c_masks.to(device),
                                    sample=True,
                                    # strict=True
                                    )
            cds_output_str = [ ''.join([codon_vocab_reverse[a] for a in _tokens]) for _tokens in output_tokenized]


    with torch.no_grad():
        # # # sample for UTR
        inp = torch.tensor(tokenizer.tokenize_to_ids(prt, seq_type='aa') + tokenizer.encode([start_tokens[gen_type]])).long()
        with torch.cuda.amp.autocast():
            utr_scores, sample = model.spt.generate(max_new_tokens, inp[None, ...], eos_token=0, temperature=temperature, retrun_score=True)
            output_tokenized = sample[0].cpu().tolist()
            utr_output_str = ''.join(tokenizer.decode([s for s in output_tokenized]))


    outputs['output'] = [cds_output_str[0], cds_scores.item(), utr_output_str, utr_scores.item()]
    outputs["num_tokens"] = len(output_tokenized)

    print(f'Process {accelerator.process_index} generate output: {outputs}')
  
    return outputs


def generator(model, tokenizer, prompts, predict_func, post_func, args):
    """
    generator for parallel prediction on given model

    """


    accelerator = args.accelerator

    device = accelerator.device
    accelerator.wait_for_everyone()
    start=time.time()

   # prepare for accelerator
    model, prompts= accelerator.prepare(
        model, prompts
    )

    
        # log
    if accelerator.is_main_process:
        logger.info(f'''***** Running training *****:
            Num examples:             {len(prompts)}
            Model checkpoint:               {args.init_from_checkpoint}
            Max seq lenghth:         {args.max_seq_len}
            Device Num:          {device}
            GPU Nums:            {accelerator.num_processes}
            Model output dir：    {args.output_dir}
        ''')


    with accelerator.split_between_processes(prompts) as prompts_all:
 
        results=dict(outputs=[], num_tokens=0)
        for prompt in prompts_all:
            outputs = predict_func(accelerator.unwrap_model(model), tokenizer, prompt, accelerator)
            results["outputs"].append(outputs)
            # logger.info(f'Process {accelerator.process_index}, generate output: {outputs}', main_process_only=False)
            results["num_tokens"] += outputs['num_tokens']
    
        timediff=time.time()-start
        logger.info("GPU {}: {} prompts received, generated {} tokens in {} seconds, {} t/s".format(
            accelerator.process_index,
            len(prompts_all),
            results["num_tokens"],
            timediff,
            results["num_tokens"]//timediff,
            ), main_process_only=False)

        results=results["outputs"]


    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])

        print(f"tokens/sec: {num_tokens//timediff}, total tokens {num_tokens}, time {timediff}")

        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            result_f = os.path.join(args.output_dir, "prediction.json")
            with open(result_f, "w") as f:
                json.dump(results_gathered, f)

            logger.info(f"save prediction to {result_f}")


def predict(args):


    dataset_name = 'data/processed/hf_datasets/genecode_human_annotation_dominant_transcript_cdna'
    dataset = load_from_disk(dataset_name)
    dataset = dataset.filter(lambda example: len(example["cds"]) % 3 == 0)

    dataset_df = dataset['test'].to_pandas()

    prompts = []
    for temperature in np.arange(0.8, 0.9, 0.1): # (0.1, 2, 0.1):
        for prt in dataset_df['aa'].head(2).values:
            for gen_type in ['mRNA']: # for different tasks
                prompts.append([prt, gen_type, temperature])


    # print(f"prompts 1: {prompts[0]}")

    exp_prompts = pd.read_excel('data/processed/全序列优化_第二轮_多肽模版_08232024.xlsx')
    prompts = []
    # for temperature in np.arange(0.2, 0.3, 0.1): 
    for temperature in [0.2]:
        for prt in exp_prompts['Peptide'].head(1).values:
            for gen_type in ['mRNA']: # for different tasks
                prompts.append([prt, gen_type, temperature])

    # prompts = [['MDVGLQRDEDDAPLCEDVELQDGDLSPE*', 'mRNA', 0.1]]
    num_return_sequences =  40
    prompts = prompts * num_return_sequences * 10

    print(f"Total samples {len(prompts)}")


    model =  SPT_CRF(
        tagset_size = 68,  # condon_vocab size
        hidden_dim = 2048,
        # model_weights =  # 'checkpoints/finetune001_t1/pytorch_model.bin'
    )
 
    if args.init_from_checkpoint:
        print(f"Loading pretrain checkpoint from {args.init_from_checkpoint}")
        model.load_state_dict(torch.load(args.init_from_checkpoint))
    
    # tokenizer
    tokenizer = Tokenizer()
    # train
    generator(model, tokenizer, prompts, predict_func, post_func=None, args=args)
