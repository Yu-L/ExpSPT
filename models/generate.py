
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

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn import functional as F
from transformers import default_data_collator, get_linear_schedule_with_warmup

from dataset.tokenizer import create_vocabulary, Vocabulary, codon_vocab, Tokenizer
from models import SPT
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
}
max_new_tokens = 4096

def predict_func(model, tokenizer, prompt, accelerator):

    prt, gen_type, temperature = prompt
    outputs = {
        'prompt': prt,
        'gen_type': gen_type,
        'temperature':temperature
    }

    inp = torch.tensor(tokenizer.tokenize_to_ids(prt, seq_type='aa') + tokenizer.encode([start_tokens[gen_type]])).long()
    with torch.cuda.amp.autocast():
        sample = model.generate(max_new_tokens, inp[None, ...], eos_token=0, temperature=temperature)
        output_tokenized = sample[0].cpu().tolist()
        output_str = ''.join(tokenizer.decode([s for s in output_tokenized]))

    outputs['output'] = output_str
    outputs["num_tokens"] = len(output_tokenized)
    # print(f'Process generate output: {outputs}', main_process_only=False)
    print(f'Process {accelerator.process_index} generate output: {outputs}')
    # accelerator.print(f'Process {accelerator.process_index} generate output: {outputs}')
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


    dataset_name = 'data/processed/hf_datasets/paired_utr3_preference_rank1.5_roughly_diff' #'data/processed/hf_datasets/genecode_human_annotation_dominant_transcript_cdna'
    dataset = load_from_disk(dataset_name)
    # dataset = dataset.filter(lambda example: len(example["cds"]) % 3 == 0)


    dataset_df = dataset['train'].to_pandas()

    prompts = []
    for temperature in np.arange(0.1, 0.5, 0.1): # (0.1, 2, 0.1):
        for prt in dataset_df['prompt'].values: #dataset_df['aa'].values:
            for gen_type in ['UTR3']: # start_tokens.keys(): # for different tasks
                prompts.append([prt, gen_type, temperature])

    print(f"Total samples {len(prompts)}")
    # model 
    model = SPT(
        num_tokens= 101,
        dim=2048,
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

    # train
    generator(model, tokenizer, prompts, predict_func, post_func=None, args=args)
