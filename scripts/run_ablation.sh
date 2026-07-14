#!/bin/bash

# ================= 配置区域 =================
export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
MASTER_PORT=29688  # 端口换个新的防止冲突

# 注意：这些路径是相对于项目根目录的
DATA_PATH="datasets/train_ForenSynths/train"
EVAL_PATH="datasets/test3_Ojha/test"

DIR_BASELINE="results/ablation/baseline_hh_only"
DIR_NAIVE="results/ablation/naive_concat"
DIR_OURS="results/ablation/ours_attention" 

echo "========================================================"
echo "🧪 启动全套消融实验 (Baseline -> Naive -> Ours)"
echo "========================================================"

# --------------------------------------------------------
# [1/3] Baseline (HH Only)
# --------------------------------------------------------
echo -e "\n➡️  [1/3] Running Baseline (HH)..."
if [ ! -d "$DIR_BASELINE" ]; then
    torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT main_finetune_mb.py \
        --model SAFE_MB --ablation_mode baseline \
        --batch_size 64 --epochs 30 \
        --data_path $DATA_PATH --eval_data_path $EVAL_PATH \
        --output_dir $DIR_BASELINE
fi
# 评估
torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT main_finetune_mb.py \
    --model SAFE_MB --ablation_mode baseline \
    --resume "$DIR_BASELINE/checkpoint-best.pth" \
    --eval true --eval_data_path $EVAL_PATH

# --------------------------------------------------------
# [2/3] Naive MB (Concat)
# --------------------------------------------------------
echo -e "\n➡️  [2/3] Running Naive MB (Concat)..."
if [ ! -d "$DIR_NAIVE" ]; then
    torchrun --nproc_per_node=$GPU_NUM --master_port=$(($MASTER_PORT + 1)) main_finetune_mb.py \
        --model SAFE_MB --ablation_mode naive \
        --batch_size 64 --epochs 30 \
        --data_path $DATA_PATH --eval_data_path $EVAL_PATH \
        --output_dir $DIR_NAIVE
fi
# 评估
torchrun --nproc_per_node=$GPU_NUM --master_port=$(($MASTER_PORT + 1)) main_finetune_mb.py \
    --model SAFE_MB --ablation_mode naive \
    --resume "$DIR_NAIVE/checkpoint-best.pth" \
    --eval true --eval_data_path $EVAL_PATH

# --------------------------------------------------------
# [3/3] Ours (Attention) <--- 最关键的 88.25% 模型
# --------------------------------------------------------
echo -e "\n➡️  [3/3] Running Ours (Attention)..."
if [ ! -d "$DIR_OURS" ]; then
    torchrun --nproc_per_node=$GPU_NUM --master_port=$(($MASTER_PORT + 2)) main_finetune_mb.py \
        --model SAFE_MB --ablation_mode ours \
        --batch_size 64 --epochs 30 \
        --data_path $DATA_PATH --eval_data_path $EVAL_PATH \
        --output_dir $DIR_OURS \
        --dist_eval true 
fi
# 评估
torchrun --nproc_per_node=$GPU_NUM --master_port=$(($MASTER_PORT + 2)) main_finetune_mb.py \
    --model SAFE_MB --ablation_mode ours \
    --resume "$DIR_OURS/checkpoint-best.pth" \
    --eval true --eval_data_path $EVAL_PATH

echo "🎉 所有实验完成！请查看 results/ablation 下的结果。"
