#!/bin/bash
# 修正版：伪装单卡DDP，暂时只跑1个种子抢核心数据

SEEDS=(42)
MODELS=("naive" "ours")

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "🚀 正在运行: 模型=$model, 随机种子=$seed, Epochs=20 (GPU: TITAN Xp)"
        
        # 使用 torchrun 启动，伪装成 1个节点 1张卡 的集群，解决 all_gather 崩溃问题
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29500 main_finetune_mb.py \
            --model SAFE_MB \
            --ablation_mode $model \
            --seed $seed \
            --batch_size 32 \
            --epochs 20 \
            --data_path datasets/train_ForenSynths \
            --eval_data_path datasets/test3_Ojha/test \
            --output_dir results/ablation_seeds/${model}_seed${seed} 
    done
done
echo "✅ 单一种子双模型跑通，请查收结果！"
