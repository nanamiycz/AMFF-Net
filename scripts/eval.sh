#!/bin/bash

# ================= 配置区域 =================

# [硬件适配] 强制使用 GPU 1 (TITAN Xp)
GPU_NUM=1
export CUDA_VISIBLE_DEVICES=1

# 分布式端口
MASTER_ADDR=localhost
MASTER_PORT=29506

# 模型名称
MODEL="SAFE"

# ==========================================================
# [关键修改] 使用你刚刚找到的确切路径
# ==========================================================
CHECKPOINT_DIR="results/SAFE_reproduction/20260112_151932_SAFE_Reproduction"
RESUME_FILE="checkpoint/checkpoint-best.pth"

echo "--------------------------------------------------------"
echo "Target Checkpoint: $RESUME_FILE"
echo "--------------------------------------------------------"

# 检查文件是否存在
if [ ! -f "$RESUME_FILE" ]; then
    echo "Error: Checkpoint file not found at $RESUME_FILE"
    exit 1
fi

# 数据集路径 (只保留你拥有的 test1, test2, test3)
eval_datasets=(
    "datasets/test1_ForenSynths/test" \
    "datasets/test2_Self-Synthesis/test" \
    "datasets/test3_Ojha/test"
)

# ================= 评估循环 =================

for eval_dataset in "${eval_datasets[@]}"
do
    if [ ! -d "$eval_dataset" ]; then
        echo "Warning: Dataset not found: $eval_dataset, skipping..."
        continue
    fi

    echo "Evaluating on: $eval_dataset"

    torchrun --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_finetune.py \
        --model $MODEL \
        --input_size 256 \
        --transform_mode 'crop' \
        --eval_data_path "$eval_dataset" \
        --batch_size 128 \
        --num_workers 8 \
        --output_dir "$CHECKPOINT_DIR" \
        --resume "$RESUME_FILE" \
        --eval true \
        --dist_eval true \
        --pin_mem true

done