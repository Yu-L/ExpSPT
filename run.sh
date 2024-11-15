

WORLD_SIZE=1
RANK=0
NUM_PROC=8

# NOTE: --multi_gpu 强制使用pytorch原生分布式而非deepspeed
############################################
### 运行案例
############################################

############################################
# echo -e "run pretrain ... "
# # export NCCL_DEBUG=info
# # _with_deepspeed
# accelerate launch \
#         --config_file configs/accelerate_deepspeed_config.yaml \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes ${NUM_PROC} \
#         main.py \
#             --type pretrain \
#             --mode train \
#             --dataset_name 'data/processed/hf_datasets/RNA_large_pretrain_processed' \
#             --model_name "pretrain004_t1" \
#             --resume_from_checkpoint 'checkpoints/pretrain7B_001_t1/step_200' \
#             --learning_rate 3e-5 \
#             --weight_decay 0.001 \
#             --clip_grad 0.0 \
#             --gradient_accumulation_steps 16 \
#             --num_warmup_steps 50 \
#             --per_device_train_batch_size 2 \
#             --per_device_eval_batch_size 2 \
#             --num_train_epochs 1 \
#             --preprocessing_num_workers 64 \
#             --checkpointing_steps 100 \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/pretrain004_t1.log 


# ##############################################

# # set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144
echo -e "run downstream tasks ... "
export NCCL_DEBUG=debug # info
export NCCL_BLOCKING_WAIT=0
accelerate launch \
        --config_file configs/accelerate_deepspeed_config.yaml \
        --num_machines $WORLD_SIZE \
        --machine_rank $RANK \
        --num_processes $NUM_PROC \
        --multi_gpu \
        main.py \
            --type task \
            --mode train \
            --regression True \
            --dataset_name 'data/processed/hf_datasets/MRL_egfp_unmod_1' \
            --model_name "MRL_egfp_unmod_1_001" \
            --init_from_checkpoint 'checkpoints/task_sft/regression/pretrain001_t5_5000_state/pytorch_model.bin' \
            --learning_rate 7e-07 \
            --weight_decay 0.001 \
            --beta 0.3 \
            --clip_grad 0.5 \
            --num_train_epochs 1 \
            --gradient_accumulation_steps 8 \
            --num_warmup_steps 5 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --preprocessing_num_workers 64 \
            --checkpointing_steps 100 \
            --with_tracking \
            --report_to "tensorboard" \
        2>&1 | tee logs/MRL_egfp_unmod_1_001.log 


##############################################
# echo -e "run finetune ... "
# # export NCCL_DEBUG=info
# accelerate launch \
#  		--multi_gpu \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes 8 \
#         --config_file configs/accelerate_with_deepspeed_config.yaml \
#         main.py \
#             --type finetune \
#             --mode train \
#             --dataset_name 'data/processed/hf_datasets/RNA_finetune_flattened_large' \
#             --model_name "finetune004_t1" \
#             --resume_from_checkpoint 'checkpoints/finetune/step_1' \
#             --learning_rate 3e-5 \
#             --weight_decay 0.001 \
#             --clip_grad 0.0 \
#             --gradient_accumulation_steps 8 \
#             --num_warmup_steps 5 \
#             --per_device_train_batch_size 1 \
#             --per_device_eval_batch_size 1 \
#             --num_train_epochs 5 \
#             --preprocessing_num_workers 64 \
#             --checkpointing_steps 100 \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/finetune004_t1.log 

##############################################

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144
# echo -e "run CRF ... "
# export NCCL_DEBUG=info
# accelerate launch \
#  		--multi_gpu \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes $NUM_PROC \
#         --config_file configs/accelerate_config.yaml \
#         main.py \
#             --type crf \
#             --mode train \
#             --dataset_name 'data/processed/hf_datasets/all_cells_omics_high_mRNA_processed1' \
#             --model_name "CRF003_t1" \
#             --resume_from_checkpoint 'checkpoints/CRF_finetune/step_2' \
#             --learning_rate 3e-6 \
#             --weight_decay 0.001 \
#             --gradient_accumulation_steps 8 \
#             --num_warmup_steps 5 \
#             --per_device_train_batch_size 1 \
#             --per_device_eval_batch_size 1 \
#             --num_train_epochs 5 \
#             --preprocessing_num_workers 64 \
#             --checkpointing_steps 100 \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/CRF003_t1.log 


# ##############################################

# # set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144
# # echo -e "run RF DPO ... "
# # export NCCL_DEBUG=info
# accelerate launch \
#  		--multi_gpu \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes $NUM_PROC \
#         --config_file configs/accelerate_config.yaml \
#         main.py \
#             --type mdpo \
#             --mode train \
#             --dataset_name 'data/processed/hf_datasets/paired_utr3_preference_rank1.5_roughly_diff' \
#             --model_name "UTR3_DPO" \
#             --init_from_checkpoint 'checkpoints/finetune001_t1/pytorch_model.bin' \
#             --resume_from_checkpoint 'checkpoints/UTR3_DPO/step_0_checkpoint' \
#             --learning_rate 7e-07 \
#             --weight_decay 0.001 \
#             --beta 0.3 \
#             --clip_grad 0.5 \
#             --num_train_epochs 2 \
#             --gradient_accumulation_steps 8 \
#             --num_warmup_steps 5 \
#             --per_device_train_batch_size 1 \
#             --per_device_eval_batch_size 1 \
#             --preprocessing_num_workers 64 \
#             --checkpointing_steps 100 \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/UTR3_DPO001_t1.log 


##############################################
# 分布式DPO强化学习
# python main.py \
#             --type dpo \
#             --mode search \
#             --dataset_name 'data/processed/hf_datasets/paired_utr3_preference_rank1.5_roughly_diff' \
#             --model_name "UTR3_DPO" \
#             --init_from_checkpoint 'checkpoints/finetune001_t1/pytorch_model.bin' \
#             --resume_from_checkpoint 'checkpoints/UTR3_DPO/step_0_checkpoint' \
#             --learning_rate 7e-07 \
#             --weight_decay 0.001 \
#             --beta 0.3 \
#             --clip_grad 0.5 \
#             --num_train_epochs 2 \
#             --gradient_accumulation_steps 8 \
#             --num_warmup_steps 5 \
#             --per_device_train_batch_size 1 \
#             --per_device_eval_batch_size 1 \
#             --preprocessing_num_workers 64 \
#             --checkpointing_steps 100 \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/UTR3_DPO001_t1.log 

############################################
# echo -e "run predict ... "
# # export NCCL_DEBUG=info
# accelerate launch \
#  		--multi_gpu \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes 8 \
#         --config_file configs/accelerate_with_deepspeed_config.yaml \
#         main.py \
#             --type predict \
#             --mode predict \
#             --dataset_name 'data/processed/hf_datasets/genecode_human_annotation_dominant_transcript_cdna' \
#             --init_from_checkpoint 'checkpoints/paired_mRNA_preference_rank1.5_roughly_diff_dpo06_T7/step_1800_checkpoint/pytorch_model.bin' \
#             --model_name "predict001_t1" \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/predict.log 

##############################################
# # 全序列批量生成
# echo -e "run predict for whole mRNA ... "
# # export NCCL_DEBUG=info
# accelerate launch \
#  		--multi_gpu \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes 4 \
#         --config_file configs/accelerate_with_deepspeed_config.yaml \
#         main.py \
#             --type predict \
#             --mode generate \
#             --dataset_name 'data/processed/hf_datasets/genecode_human_annotation_dominant_transcript_cdna' \
#             --init_from_checkpoint 'checkpoints/CRF003_t1/best_checkpoint/pytorch_model.bin' \
#             --model_name "predict001_t1.20" \
#             --with_tracking \
#             --report_to "tensorboard" \
#         2>&1 | tee logs/predict001_t1.20.log 

##############################################

############################################################


## PAI灵骏智算平台/任务 例子
# cd /maindata/data/user/user_wan/yul/bioSPT2/
# accelerate launch \
#          --main_process_ip $MASTER_ADDR \
#         --main_process_port $MASTER_PORT \
#         --num_machines $WORLD_SIZE \
#         --machine_rank $RANK \
#         --num_processes 32 \
#          --config_file configs/accelerate_with_deepspeed_config.yaml \
# train_dps.py \
#     --dataset_name 'data/processed/hf_datasets/RNA_large_pretrain_processed' \
#     --model_name "pretrain002_t3" \
#     --learning_rate 3e-4 \
#     --weight_decay 0.001 \
#     --gradient_accumulation_steps 4 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --num_train_epochs 3 \
#     --preprocessing_num_workers 64 \
#     --checkpointing_steps 100 \
#     --with_tracking \
#     --report_to "tensorboard" \
# 2>&1 | tee logs/pretrain002_t3.log 


###########################################################



