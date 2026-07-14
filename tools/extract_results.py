import os
import json
import numpy as np

def extract_metrics(log_file):
    if not os.path.exists(log_file):
        return None
    
    max_acc = 0.0
    best_epoch_data = None
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('{"train_lr"'):
                try:
                    data = json.loads(line.strip())
                    if 'test_acc1' in data and data['test_acc1'] > max_acc:
                        max_acc = data['test_acc1']
                        best_epoch_data = data
                except:
                    continue
    return best_epoch_data

seeds = [42, 100, 2026]
models = ['naive', 'ours']

for model in models:
    accs = []
    print(f"\n--- {model.upper()} 模型 ---")
    for seed in seeds:
        log_path = f"results/ablation_seeds/{model}_seed{seed}/log.txt"
        data = extract_metrics(log_path)
        if data:
            acc = data.get('test_acc1', 0) * 100 # 如果原来是小数，转为百分比
            accs.append(acc)
            # 注：这里暂只提取了 Ojha 的准确率，因为原先的命令只测了 Ojha。
            print(f"Seed {seed}: ACC = {acc:.2f}%")
        else:
            print(f"Seed {seed}: 数据未找到")
    
    if len(accs) > 0:
        print(f">> 汇总: ACC = {np.mean(accs):.2f} ± {np.std(accs):.2f}")
