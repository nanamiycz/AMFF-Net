#!/bin/bash
SEEDS=(42 100 2026)
MODELS=("naive" "ours")

# 【已修正】换成您真实的文件夹路径
DATASETS=(
    "datasets/test1_ForenSynths/test" 
    "datasets/test3_Ojha/test" 
    "datasets/test2_Self-Synthesis/test" 
)

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CKPT="results/ablation_seeds/${model}_seed${seed}/checkpoint-19.pth"
        if [ ! -f "$CKPT" ]; then
            CKPT="results/ablation_seeds/${model}_seed${seed}/clean_weight.pth"
        fi

        for data_path in "${DATASETS[@]}"; do
            echo "--------------------------------------------------------"
            echo "🔍 评估: 模型 [ ${model} ] | 种子 [ ${seed} ] | 数据集 [ ${data_path} ]"
            
            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 main_finetune_mb.py \
                --model SAFE_MB \
                --ablation_mode $model \
                --eval True \
                --resume $CKPT \
                --eval_data_path $data_path \
                --batch_size 64 
        done
    done
done
echo "✅ 全家桶跑分真正完毕！"
