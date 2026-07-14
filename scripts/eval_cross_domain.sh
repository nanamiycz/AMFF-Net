#!/bin/bash

# ================= 配置区域 =================
GPU_NUM=1
MASTER_ADDR=localhost
MASTER_PORT=29614  # 换个新端口，绝对不冲突

# ================= 自动寻找最新的 Diffusion 权重 =================
# [关键修改 1] 把寻找路径改成了刚才跑出来的 diffusion_trained_model
LATEST_DIR=$(ls -td results/diffusion_trained_model/*SAFE* | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 错误: 未找到 diffusion 训练的权重！请确认上一步跑成功了。"
    exit 1
fi

LATEST_DIR=${LATEST_DIR%/}
CHECKPOINT="${LATEST_DIR}/checkpoint-best.pth"
# [关键修改 2] 给日志换个名字，叫交叉评估日志
LOG_FILE="${LATEST_DIR}/cross_eval_log.txt"

echo "========================================================"
echo "🎯 反向交叉验证评估 (Diffusion训练 -> 测试GANs)"
echo "📂 权重路径: $CHECKPOINT"
echo "📝 结果日志: $LOG_FILE"
echo "========================================================"

# ================= 定义测试集 =================
eval_datasets=(
    "datasets/test1_ForenSynths/test"    # [核心目标] 用来测 ProGAN, StyleGAN 等旧假图
    "datasets/test3_Ojha/test"           # [顺手测测] 看看在自己老家 Diffusion 上的表现
)

# 清空旧日志
echo "Cross Domain Evaluation Report - $(date)" > "$LOG_FILE"

# ================= 循环评估 =================
for eval_dataset in "${eval_datasets[@]}"
do
    if [ ! -d "$eval_dataset" ]; then
        echo "⚠️  跳过: 路径不存在 -> $eval_dataset" | tee -a "$LOG_FILE"
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    echo "🚀 正在启动评估: $eval_dataset" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    
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
    sleep 2
done

echo ""
echo "🎉 反向交叉评估完成！"
echo "👉 请打开文件查看结果填表: $LOG_FILE"