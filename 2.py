import matplotlib.pyplot as plt
import numpy as np

# 您的数据
noise = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

# Deepfake 数据
df_base = [85.22, 30.00, 15.78, 15.41, 15.41, 15.41]
df_naive = [85.06, 16.66, 16.12, 28.88, 84.59, 84.59]

# Diffusion 数据
diff_base = [79.75, 53.10, 53.45, 55.45, 58.15, 61.10]
diff_naive = [76.35, 61.05, 60.50, 53.00, 50.00, 50.00]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Deepfake
ax1.plot(noise, df_base, 'r-o', label='Baseline (Ours)', linewidth=2.5)
ax1.plot(noise, df_naive, 'b--s', label='Naive (Force 1.0)', linewidth=2.5)
ax1.set_title("Deepfake (OOD): Signal Dampening", fontsize=14, fontweight='bold')
ax1.set_xlabel("Noise Level ($\sigma$)", fontsize=12)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()
ax1.annotate('Baseline Collapses\n(Dampening)', xy=(0.04, 15.78), xytext=(0.02, 40),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Plot 2: Diffusion
ax2.plot(noise, diff_base, 'r-o', label='Baseline (Ours)', linewidth=2.5)
ax2.plot(noise, diff_naive, 'b--s', label='Naive (Force 1.0)', linewidth=2.5)
ax2.set_title("Diffusion (Target): Smart Filtering", fontsize=14, fontweight='bold')
ax2.set_xlabel("Noise Level ($\sigma$)", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
ax2.annotate('Baseline Robustness\n(Filtering)', xy=(0.08, 58.15), xytext=(0.05, 70),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig('figure_double_edged_sword.png', dpi=300)
print("Plot saved as figure_double_edged_sword.png")