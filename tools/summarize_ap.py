import re
import numpy as np

def extract_from_log(log_file):
    results = {'naive': {}, 'ours': {}}
    current_model = None
    current_dataset = None
    
    with open(log_file, 'r') as f:
        for line in f:
            model_match = re.search(r'模型 \[ (naive|ours) \] \| 种子 \[ (\d+) \] \| 数据集 \[ (.*?) \]', line)
            if model_match:
                current_model = model_match.group(1)
                seed = model_match.group(2)
                dataset_path = model_match.group(3)
                
                if 'test1_ForenSynths' in dataset_path:
                    dataset_type = 'test1_ForenSynths'
                elif 'test3_Ojha' in dataset_path:
                    dataset_type = 'test3_Ojha'
                elif 'test2_Self-Synthesis' in dataset_path:
                    dataset_type = 'test2_Self-Synthesis'
                else:
                    continue
                    
                if seed not in results[current_model]:
                    results[current_model][seed] = {}
                if dataset_type not in results[current_model][seed]:
                    results[current_model][seed][dataset_type] = {'acc': [], 'ap': []}
                
                current_dataset = dataset_type
                continue
                
            acc_ap_match = re.search(r'acc: ([\d.]+)\%, ap: ([\d.]+)\%', line)
            if acc_ap_match and current_model and current_dataset:
                acc = float(acc_ap_match.group(1))
                ap = float(acc_ap_match.group(2))
                results[current_model][seed][current_dataset]['acc'].append(acc)
                results[current_model][seed][current_dataset]['ap'].append(ap)

    for model in ['naive', 'ours']:
        print(f"\n=== {model.upper()} ===")
        for dataset_type in ['test1_ForenSynths', 'test3_Ojha', 'test2_Self-Synthesis']:
            all_accs = []
            all_aps = []
            for seed in results[model]:
                if dataset_type in results[model][seed] and results[model][seed][dataset_type]['acc']:
                    seed_avg_acc = np.mean(results[model][seed][dataset_type]['acc'])
                    seed_avg_ap = np.mean(results[model][seed][dataset_type]['ap'])
                    all_accs.append(seed_avg_acc)
                    all_aps.append(seed_avg_ap)
            
            if all_accs:
                print(f"{dataset_type}: ACC {np.mean(all_accs):.2f}±{np.std(all_accs):.2f} / AP {np.mean(all_aps):.2f}±{np.std(all_aps):.2f}")

extract_from_log('final_eval_results.log')
