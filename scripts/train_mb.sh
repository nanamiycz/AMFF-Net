#!/bin/bash

# ================= 配置区域 =================

# [单卡设置 - 彻底屏蔽那张老的 TITAN Xp，只用新显卡]
GPU_NUM=1
export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=localhost
MASTER_PORT=29610 # 换个全新端口防冲突

# [关键修改 1] 数据集路径：指向你服务器上已有的 Ojha 扩散模型数据
train_datasets=("datasets/test3_Ojha/test/guided")

# ==========================================================
# 指定新模型名 (对应 main_finetune_mb.py 里的注册名)
# ==========================================================
MODEL="SAFE_MB"

# ================= 训练循环 =================
for train_dataset in "${train_datasets[@]}" 
do
    current_time=$(date +"%Y%m%d_%H%M%S")
    
    # [关键修改 2] 输出目录修改为 diffusion_trained_model，保护以前的权重
    OUTPUT_PATH="results/diffusion_trained_model/${current_time}_SAFE_MultiBand_SAFE_MB_Attention"
    mkdir -p $OUTPUT_PATH

    echo "========================================================"
    echo "Starting INNOVATION Training: Multi-Band Frequency Fusion"
    echo "Model: $MODEL | Script: main_finetune_mb.py"
    echo "Dataset: $train_dataset"
    echo "========================================================"
    
    # 调用 python 入口文件 main_finetune_mb.py
    torchrun --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_finetune_mb.py \
        --model $MODEL \
        --input_size 256 \
        --transform_mode 'crop' \
        --data_path "$train_dataset" \
        --eval_data_path "$train_dataset" \
        --save_ckpt_freq 5 \
        --batch_size 32 \
        --update_freq 2 \
        --blr 1e-2 \
        --weight_decay 0.01 \
        --warmup_epochs 1 \
        --epochs 20 \
        --num_workers 8 \
        --output_dir $OUTPUT_PATH \
        --pin_mem true \
        --dist_eval true \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt
done