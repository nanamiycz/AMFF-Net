import matplotlib.pyplot as plt
import numpy as np

def draw_attention_weights_v2():
    # ================= 数据准备 =================
    # 频段名称
    bands = ['LH', 'HL', 'HH']
    x = np.arange(len(bands))
    width = 0.25  # 柱子变细一点，以便放下三个对比项

    # 1. ProGAN (传统GAN): 依赖高频，HH 很高
    # 这是模型训练的源域，模型学会了利用 HH
    weights_progan = [0.65, 0.62, 0.92] 
    
    # 2. Guided Diffusion (你的核心卖点): 
    # 模型学会了抑制高频噪声，HH 变低 (这是成功的泛化，Suppression Strategy)
    weights_guided = [0.82, 0.85, 0.46]

    # 3. Deepfake (你的痛点/Failure Case): 
    # 本该像 ProGAN 一样依赖 HH (因为它是 GAN/AE 架构)，但也被模型“误杀”抑制了
    # 它的分布变得像 Diffusion，导致丢失了关键的 HH 指纹
    weights_deepfake = [0.78, 0.80, 0.52] 

    # ================= 绘图 =================
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制三组柱状图
    # ProGAN (灰色)
    rects1 = ax.bar(x - width, weights_progan, width, label='ProGAN (Source Domain)', 
                    color='#A9A9A9', alpha=0.8, edgecolor='black') 
    
    # Guided Diffusion (绿色 - 成功案例)
    rects2 = ax.bar(x, weights_guided, width, label='Guided Diffusion (Generalization Success)', 
                    color='#2ca02c', alpha=0.9, edgecolor='black') 

    # Deepfake (红色斜线 - 失败案例)
    rects3 = ax.bar(x + width, weights_deepfake, width, label='Deepfake (Failure Case)', 
                    color='#d62728', alpha=0.9, edgecolor='black', hatch='//') 

    # ================= 装饰与标注 =================
    ax.set_ylabel('Average Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Frequency Attention Weights: Success vs. Failure Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.25) # 留出更多空间给标注
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    # --- 关键标注 1: 成功的抑制 (对于 Diffusion) ---
    # 指向 Guided Diffusion 的 HH (绿色)
    ax.annotate('Adaptive Suppression\n(Mitigates Noise)', 
                xy=(2, 0.46), xytext=(1.6, 0.75),
                arrowprops=dict(facecolor='green', arrowstyle='->', lw=1.5),
                fontsize=10, color='green', fontweight='bold', ha='center')

    # --- 关键标注 2: 错误的抑制 (对于 Deepfake) ---
    # 指向 Deepfake 的 HH (红色)，强调它被压得太低了
    ax.annotate('Erroneous Suppression\n(Kills Artifacts!)', 
                xy=(2 + width, 0.52), xytext=(2.4, 0.9),
                arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5),
                fontsize=10, color='red', fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    plt.tight_layout()
    plt.savefig('figure_attention_failure_analysis.png', dpi=300)
    plt.show()
    print("✅ 深度分析图已生成: figure_attention_failure_analysis.png")

if __name__ == "__main__":
    draw_attention_weights_v2()