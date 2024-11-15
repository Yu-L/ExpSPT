from models.pretrain import train as pretrain
from models.task_sft import train as SFTtrain, main as SFTmain
from models.dpo import train as DPOtrain, main as DPOmain
from models.dpo_dps import train as mDPOtrain, main as mDPOmain
from models.train_sft import train as finetune_train
from models.crf_sft import train as CRFtrain
from models.generate import predict 
from models.predict import predict  as generate
from transformers import SchedulerType,  get_scheduler
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
import argparse
from datetime import timedelta
from accelerate.logging import get_logger
from datetime import datetime
from pytz import timezone
import time
import logging


def timetz(*args):
    return datetime.now(tz).timetuple()

tz = timezone('Asia/Shanghai') # UTC, Asia/Shanghai, Europe/Berlin



def parse_args():
    parser = argparse.ArgumentParser(description="biospt2")
    
    parser.add_argument('--type', type=str, required=True, help='sft | dpo')
    parser.add_argument('--mode', type=str, required=True, help='only train | parameter gridsearch')

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='model',
        help="identier for model.",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--regression",
        type=bool,
        default=False,
        help="if a regression task",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--clip_grad", type=float, default=0.0, help="clip grad to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--grad_norm",
        type=float,
        default=0.5,
        help="grad_norm.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="dpo beta.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=None,
        help="If the training should start from a checkpoint file.",
    )
    # New Code #
    # Whether to load the best model at the end of training
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"`, and `"dvclive"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    
    parser.add_argument(
        "--accelerator",
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"`, and `"dvclive"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need either a dataset name or a training/validation file.")

    return args



def main():

    args = parse_args()

    args.output_dir = f"checkpoints/{args.model_name}"
    # accelerator
    accelerator = (
        Accelerator(
            log_with='tensorboard',
            project_dir=args.output_dir,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100 * 3600))]
        )
        if args.with_tracking
        else Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    )
    logger = get_logger(__name__)
   # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.Formatter.converter = timetz

    args.accelerator = accelerator
    args.logger = logger

    if args.type == 'pretrain':
        if args.mode == 'train':
            pretrain(args)

    if args.type == 'finetune':
        if args.mode == 'train':
            finetune_train(args)
    
    # 下游任务
    if args.type == 'task':
        if args.mode == 'train':
            SFTtrain(args)
        if args.mode == 'search':
            SFTmain()

    # CRF 任务微调
    if args.type == 'crf':
        if args.mode == 'train':
            CRFtrain(args)

    if args.type == 'dpo':
        if args.mode == 'train':
            DPOtrain()
        if args.mode == 'search':
            DPOmain()
    # 分布式训练DPO
    if args.type == 'mdpo':
        if args.mode == 'train':
            mDPOtrain(args=args)
        if args.mode == 'search':
            mDPOmain()

    #　推理生成
    if args.type == 'predict':
        if args.mode == 'predict':
            predict(args)
        if args.mode == 'generate':
            generate(args)

        

            

if __name__ == '__main__':
    main()

        