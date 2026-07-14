import re
import numpy as np

log_file = 'table6_results.log'
results = {}
current_weight = None
current_domain = None

with open(log_file, 'r') as f:
    for line in f:
        # 匹配权重和数据集类型
        m_head = re.search(r'正在测试权重: \[ (.*?) \].*?数据集: \[ (.*?) \]', line)
        if m_head:
            current_weight = m_head.group(1)
            ds_path = m_head.group(2)
            current_domain = 'Diffusion (Ojha)' if 'test3_Ojha' in ds_path else 'Legacy (Self-Synthesis)'
            
            if current_weight not in results:
                results[current_weight] = {}
            if current_domain not in results[current_weight]:
                results[current_weight][current_domain] = {'acc': [], 'ap': []}
            continue
            
        # 匹配具体的 acc 和 ap
        m_metrics = re.search(r'acc: ([\d.]+)\%, ap: ([\d.]+)', line)
        if m_metrics and current_weight and current_domain:
            acc = float(m_metrics.group(1))
            ap = float(m_metrics.group(2))
            results[current_weight][current_domain]['acc'].append(acc)
            results[current_weight][current_domain]['ap'].append(ap)

print("\n" + "="*50)
print("📊 Table 6 真实跑分均值汇总 📊")
print("="*50)
for weight in results:
    print(f"\n🚀 【{weight}】")
    for domain in results[weight]:
        accs = results[weight][domain]['acc']
        aps = results[weight][domain]['ap']
        if accs:
            print(f"  👉 {domain} (包含 {len(accs)} 个子数据集):")
            print(f"     平均 ACC: {np.mean(accs):.2f}%")
            print(f"     平均 AP : {np.mean(aps):.2f}%")
print("\n" + "="*50)
