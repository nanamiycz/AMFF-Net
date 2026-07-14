#!/bin/bash
MODELS=("weight_naive.pth" "weight_kd.pth")
DATASETS=("datasets/test3_Ojha/test" "datasets/test2_Self-Synthesis/test")

for weight in "${MODELS[@]}"; do
    for data_path in "${DATASETS[@]}"; do
        echo "========================================================"
        echo "🔍 正在测试权重: [ ${weight} ] | 数据集: [ ${data_path} ]"
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29509 main_finetune_mb.py \
            --model SAFE_MB --ablation_mode ours --eval True \
            --resume $weight --eval_data_path $data_path --batch_size 64 
    done
done
