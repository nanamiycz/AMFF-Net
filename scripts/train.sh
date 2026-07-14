#!/bin/bash

# ================= 可外部传参的配置区域 =================
# 用法：bash train.sh [模型名] [输出目录] [训练集后缀] [训练轮数]
# 示例：bash train.sh resnet18 results/ablation/baseline_spatial progan 30

# 1. 外部传参（无传参则用默认值）
MODEL=${1:-"resnet18"}          # 模型名：resnet18/SAFE
OUTPUT_DIR=${2:-"results/ablation/baseline_spatial"}  # 输出目录
TRAIN_SET_SUFFIX=${3:-"progan"} # 训练集后缀：progan/ForenSynths
EPOCHS=${4:-30}                 # 训练轮数

# 2. 固定配置（和你原有逻辑一致，不改动）
GPU_NUM=2
export CUDA_VISIBLE_DEVICES=0,1
MASTER_ADDR=localhost
MASTER_PORT=29605
TRAIN_DATASET="datasets/train_${TRAIN_SET_SUFFIX}"  # 拼接训练集路径

# ================= 训练执行 =================
current_time=$(date +"%Y%m%d_%H%M%S")
# 输出路径：如果是消融实验，直接用传参的OUTPUT_DIR；否则保留原有命名
if [[ $OUTPUT_DIR == *"ablation"* ]]; then
    FINAL_OUTPUT_PATH=$OUTPUT_DIR
else
    FINAL_OUTPUT_PATH="results/SAFE_reproduction/${current_time}_SAFE_Reproduction"
fi
mkdir -p $FINAL_OUTPUT_PATH

echo "========================================================"
echo "Starting Training: MODEL = $MODEL"
echo "Config: 2 GPUs | BS 128 (Effective) | LR 1e-2"
echo "Train Set: $TRAIN_DATASET | Epochs: $EPOCHS"
echo "========================================================"

# 分布式训练命令（开放MODEL/EPOCHS/数据路径参数）
torchrun --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_finetune.py \
    --model $MODEL \
    --input_size 256 \
    --transform_mode 'crop' \
    --data_path "$TRAIN_DATASET" \
    --eval_data_path "$TRAIN_DATASET" \
    --save_ckpt_freq 5 \
    --batch_size 32 \
    --update_freq 2 \
    --blr 1e-2 \
    --weight_decay 0.01 \
    --warmup_epochs 1 \
    --epochs $EPOCHS \
    --num_workers 8 \
    --output_dir $FINAL_OUTPUT_PATH \
    --pin_mem true \
    --dist_eval true \
    2>&1 | tee -a $FINAL_OUTPUT_PATH/log_train.txt