import matplotlib.pyplot as plt
import numpy as np

# ================= 来自你原论文 Table VI 的真实数据 =================
strategies = ['Zero-Shot (Baseline)', 'Naive Fine-Tuning', 'Multi-Layer Distillation (Ours)']
diffusion_acc = [97.02, 71.26, 96.50]  # 新领域保持率
legacy_acc =    [58.13, 81.58, 87.20]  # 旧领域回升率

x = np.arange(len(strategies))
width = 0.35 

# ================= 绘图逻辑 =================
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, diffusion_acc, width, label='New Domain (Diffusion)', color='#ff7f0e', edgecolor='black')
rects2 = ax.bar(x + width/2, legacy_acc, width, label='Old Domain (Legacy Deepfake)', color='#1f77b4', edgecolor='black')

ax.axhline(y=90, color='gray', linestyle='--', alpha=0.7)
ax.text(-0.4, 91, 'Ideal Target Zone', color='gray', fontsize=10, style='italic')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Mitigating Catastrophic Forgetting via Feature Distillation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11, fontweight='bold')
ax.set_ylim(40, 105)
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('Fig6_Continual_Learning.pdf', dpi=300, bbox_inches='tight')
print("✅ 持续学习柱状图已生成: Fig6_Continual_Learning.pdf")