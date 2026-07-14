#!/bin/bash
SEEDS=(100 2026)
MODELS=("naive" "ours")

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "🚀 正在运行: 模型=$model, 随机种子=$seed, Epochs=20 (GPU: RTX 4090 D)"
        
        # 必须用 torchrun 防止 all_gather 报错，指定 GPU 0
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main_finetune_mb.py \
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
echo "✅ 所有剩余种子在 4090 D 上完美收官！"
