#!/bin/bash

# ================= 配置区域 =================
GPU_NUM=1
MASTER_ADDR=localhost
MASTER_PORT=29613  # 端口换新，防冲突

# ================= 自动寻找最新的权重 =================
LATEST_DIR=$(ls -td results/SAFE_reproduction/*SAFE* | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 错误: 未找到 SAFE 系列的训练结果！"
    exit 1
fi

LATEST_DIR=${LATEST_DIR%/}
CHECKPOINT="${LATEST_DIR}/checkpoint-best.pth"
LOG_FILE="${LATEST_DIR}/full_eval_log.txt"

echo "========================================================"
echo "🎯 全面评估: SAFE_MB (Attention Version)"
echo "📂 权重路径: $CHECKPOINT"
echo "📝 结果日志: $LOG_FILE"
echo "========================================================"

# ================= 定义测试集 (全家桶) =================
# 数组里放你要测的所有路径
eval_datasets=(
    "datasets/test3_Ojha/test"           # [关键] Diffusion, Guided, Glide
    "datasets/test1_ForenSynths/test"    # [关键] Deepfake, GANs (检查是否回升)
    "datasets/test2_Self-Synthesis/test" # [补充] 验证同分布性能 (可选)
)

# 清空旧日志
echo "Full Evaluation Report - $(date)" > "$LOG_FILE"

# ================= 循环评估 =================
for eval_dataset in "${eval_datasets[@]}"
do
    # 检查路径是否存在，防止报错退出
    if [ ! -d "$eval_dataset" ]; then
        echo "⚠️  跳过: 路径不存在 -> $eval_dataset" | tee -a "$LOG_FILE"
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    echo "🚀 正在启动评估: $eval_dataset" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    
    # 运行评估
    # 2>&1 | tee -a ... 确保错误信息也能记录
    torchrun --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_finetune_mb.py \
        --model SAFE_MB \
        --input_size 256 \
        --transform_mode 'crop' \
        --batch_size 128 \
        --num_workers 8 \
        --output_dir "$LATEST_DIR" \
        --resume "$CHECKPOINT" \
        --eval true \
        --dist_eval true \
        --eval_data_path "$eval_dataset" 2>&1 | tee -a "$LOG_FILE"
        
    echo "✅ 完成: $eval_dataset" | tee -a "$LOG_FILE"
    # 稍微暂停一下，释放显存
    sleep 2
done

echo ""
echo "🎉 全套评估完成！"
echo "👉 请打开文件查看完整结果: $LOG_FILE"