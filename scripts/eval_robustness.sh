#!/bin/bash

# ================= 配置区域 =================
GPU_NUM=1
MASTER_PORT=29616  # 换个新端口

# ================= 自动寻找最新的 ProGAN 强力权重 =================
# 强行指定今天刚跑出的最新权重
LATEST_DIR=$(ls -td results/diffusion_trained_model/*SAFE_MB_Attention* | head -1)

if [ -z "$LATEST_DIR" ]; then
    # 如果没找到，就退而求其次用根目录下的权重
    CHECKPOINT="checkpoint/checkpoint-best.pth"
else
    CHECKPOINT="${LATEST_DIR}/checkpoint-best.pth"
fi

LOG_FILE="results/robustness_eval_log.txt"

echo "========================================================"
echo "🎯 复合退化鲁棒性测试 (Robustness Evaluation)"
echo "📂 正在使用权重: $CHECKPOINT"
echo "========================================================"

# 依次测试 4 个级别的文件夹
eval_datasets=(
    "datasets/test3_Ojha/test"             # Clean (干净原图)
    "./test_data_level_1"                  # Level 1 (轻度退化)
    "./test_data_level_2"                  # Level 2 (中度退化)
    "./test_data_level_3"                  # Level 3 (重度退化)
)

echo "Robustness Evaluation Report - $(date)" > "$LOG_FILE"

for eval_dataset in "${eval_datasets[@]}"
do
    # 检查文件夹是否存在
    if [ ! -d "$eval_dataset" ]; then
        echo "⚠️  警告: 找不到文件夹 $eval_dataset，请确认你已经运行了退化脚本！" | tee -a "$LOG_FILE"
        continue
    fi

    echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "🚀 Evaluating: $eval_dataset" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
    
    torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT main_finetune_mb.py \
        --model SAFE_MB \
        --input_size 256 \
        --transform_mode 'crop' \
        --batch_size 128 \
        --resume "$CHECKPOINT" \
        --eval true \
        --dist_eval true \
        --eval_data_path "$eval_dataset" 2>&1 | tee -a "$LOG_FILE"
        
    echo "✅ 完成: $eval_dataset" | tee -a "$LOG_FILE"
    sleep 2
done

echo "🎉 鲁棒性测试全部完成！结果已保存在 $LOG_FILE"