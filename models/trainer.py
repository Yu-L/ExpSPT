
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
from lion_pytorch import Lion
from torch.nn import functional as F
from transformers import default_data_collator, get_linear_schedule_with_warmup
from transformers import SchedulerType,  get_scheduler


from models.utils import _parameter_number, param_head


from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from datetime import datetime
from pytz import timezone
import time




""" 

train for biospt project

"""

def timetz(*args):
    return datetime.now(tz).timetuple()

tz = timezone('Asia/Shanghai') # UTC, Asia/Shanghai, Europe/Berlin
logger = get_logger(__name__)



def trainer(model, train_loader, val_loader, train_func, eval_func,  args):
    """
    Trainer for parallel train model

    """

    buffer_size = int(1e3)
    accelerator = args.accelerator
    device = accelerator.device
    
    # log
    log_dir = f"logs/tensorboard/{args.model_name}"
    # summaryWriter = SummaryWriter(log_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.Formatter.converter = timetz

    logger.info(accelerator.state, main_process_only=True)
    # print("accelerator.state: ", accelerator.state)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
        
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
 

    model_sizes = _parameter_number(model)
    accelerator.wait_for_everyone()
    # data prepare
    # with accelerator.main_process_first():


    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    # print("accelerator.state.deepspeed_plugin: ", accelerator.state.deepspeed_plugin)
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        
    
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )
        

   # prepare for accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # accelerator.print('model before:', param_head(model))
    # accelerator.load_state('/maindata/data/user/user_wan/yul/bioSPT2/checkpoints/CRF_finetune/step_0')
    # accelerator.print('model after:', param_head(model))

    # NOTE: set reduce_bucket_size value in deepspeed.json to reduce_bucket_size: hidden_size * hidden_size
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    # print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if not isinstance(checkpointing_steps, int) and checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        experiment_config["accelerator"] = None
        experiment_config["logger"] = None 
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )
    
    # log
    if accelerator.is_main_process:
        logger.info(f'''***** Running training *****:
            Num examples:             {len(train_loader)}
            Max_train_steps:          {args.max_train_steps}
            Num Epochs:               {args.num_train_epochs}
            Num_update_steps_per_epoch:       {num_update_steps_per_epoch}
            Per_device_train_batch_size:      {args.per_device_train_batch_size}
            Gradient_accumulation_steps: {accelerator.gradient_accumulation_steps}
            Total train batch size (w. parallel, distributed & accumulation): {total_batch_size}
            Learning rate:   {args.learning_rate}
            Weight_decay:    {args.weight_decay}
            clip_grad:        {args.clip_grad}
            Max seq lenghth:         {args.max_seq_len}
            Model size:      {model_sizes}
            Device Num:          {device}
            GPU Nums:            {accelerator.num_processes}
            Model output dir：    {args.output_dir}
        ''')

    
    # training
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            completed_steps = resume_step
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch
            


    # update progress bar if resumed from checkpoint
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, args.num_train_epochs):
        
        model.train()
        if args.with_tracking:
            total_loss = 0
            
        # skip new `skip_first_batches` to skip the batches when resuming from ckpt
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_loader
            
        start=time.time()
        for step, batch in enumerate(active_dataloader):
            # In particular, DeepSpeed handles `gradient_accumulation` via `DeepSpeedEngine`.
            # Below, we use `accelerator.accumulate` if the user
            # wants to switch to other approaches such as plain DDP, PyTorch FSDP ...
            # This avoids having to change any code as things are all handled across different distributed setups.

            # if completed_steps>10:
            #     break
                                                                                                                                                                                                                                                                                                                                        
            model.train()
            u_loss, step_loss = 0, 0
            with accelerator.accumulate(model):

                loss = train_func(model, batch) # train for a batch
                accelerator.backward(loss)

                if args.clip_grad >0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad) 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                u_loss += loss

                # print(f"process: {accelerator.process_index} epoch: {epoch} --step {step}, loss: {loss}")
                # 梯度累积后
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    # save model
                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)
                    if completed_steps >= args.max_train_steps:
                        break

                    # update step
                    completed_steps += 1
                    # step_loss = u_loss / accelerator.gradient_accumulation_steps
                    step_loss = accelerator.reduce(loss, reduction="mean").item() / accelerator.gradient_accumulation_steps
                    u_loss = 0

                    # We keep track of the loss at each batch
                    if args.with_tracking:
                        total_loss += step_loss
                        timediff=time.time()-start
                        logger.info(f"process: {accelerator.process_index} epoch: {epoch} -- batch: {completed_steps}, lr:{scheduler.get_lr()[0]} training loss: {step_loss} --{timediff/(completed_steps+1)} s/batch")
                        if accelerator.is_main_process:
                            accelerator.log(
                                {
                                    "step_training_loss": step_loss,
                                },
                                step=completed_steps,
                            )


                    # if i % GENERATE_EVERY == 0:
                    #     model.eval()
                    #     inp = random.choice(val_smiles)[:PRIME_LENGTH]
                    #     prime = ''.join(vocabulary.decode(inp.cpu().tolist()))

                    #     with torch.cuda.amp.autocast():
                    #         sample = accelerator.unwrap_model(model).generate(GENERATE_LENGTH, inp[None, ...], eos_token=0)

                    #     output_str = ''.join(vocabulary.decode([s for s in sample[0].cpu().tolist()]))
                    #     accelerator.print(f'{datetime.now()} -- generate on batch: {i}, prime: {prime} generate: {output_str}\n')

        
        logger.info(f"Validation for epoch {epoch} ...")
        # evaluation for every epoch
        eval_metrics = eval_func(model, val_loader, accelerator, args)
        logger.info(f"Model  validation metrics: {eval_metrics} on epoch {epoch} at training step {completed_steps} ")
        if args.with_tracking:
                
            log_msg = {
                    "epoch_train_loss": total_loss / (step+1),
                    "epoch": epoch,
                    "step": completed_steps,
                    **eval_metrics
                }

            # logger.info("log_msg: ", log_msg)
            accelerator.log(
                log_msg,
                step=epoch,
            )

        metric = eval_metrics['metric']
      # Tracks the best checkpoint and best metric
        if best_metric is None or metric > best_metric :
            best_metric = metric
            best_metric_checkpoint = os.path.join(args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            logger.info(f"New best metric: {eval_metrics} at epoch {epoch}")
            logger.info(f"best_metric_checkpoint: {best_metric_checkpoint}")

        
    # Loads the best checkpoint after the training is finished
    if args.load_best_model:
        accelerator.load_state(best_metric_checkpoint)

        # Evaluates using the best checkpoint
        eval_metrics = eval_func(model, val_loader, accelerator, args)
        logger.info(f"Best model metrics: {eval_metrics}")
        # if metric != best_metric:
        #     raise AssertionError(
        #         f"Best metric {best_metric} does not match the metric {metric} of the loaded best model."
        #     )

    # save final model
    if args.output_dir is not None:
        accelerator.wait_for_everyone()

        final_checkpoint = os.path.join(args.output_dir, "final_checkpoint")
        accelerator.save_state(final_checkpoint)
        logger.info(f"Save final checkpoint : {final_checkpoint}")
     

        unwrapped_model = accelerator.unwrap_model(model)
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        final_checkpoint = os.path.join(args.output_dir, 'pytorch_model.bin')
        if accelerator.is_main_process:
            accelerator.save(model.state_dict(), final_checkpoint)
            logger.info(f"Finish training and save best model to {final_checkpoint}")

        with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f)

        