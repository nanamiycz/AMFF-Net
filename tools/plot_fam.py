import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs('TKDD_Figures', exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis('off')

# Input
ax.add_patch(patches.Rectangle((5, 10), 8, 20, facecolor='#e3f2fd', edgecolor='#1565c0', lw=1.5))
ax.text(9, 32, 'Concat Feature\n$C \\times H/2 \\times W/2$', ha='center')
ax.annotate('', xy=(18, 20), xytext=(14, 20), arrowprops=dict(arrowstyle='->', lw=1.5))

# Squeeze
ax.add_patch(patches.Circle((22, 20), 4, facecolor='white', edgecolor='black', lw=1.5))
ax.text(22, 20, 'GAP', ha='center', va='center', fontweight='bold')
ax.annotate('', xy=(32, 20), xytext=(27, 20), arrowprops=dict(arrowstyle='->', lw=1.5))

ax.add_patch(patches.Rectangle((29, 18), 3, 4, facecolor='#fff3e0', edgecolor='#ef6c00'))
ax.text(30.5, 15, '$1 \\times 1 \\times C$', ha='center', fontsize=9)

# Excitation (FIXED: 这里换成了 FancyBboxPatch 来支持圆角)
ax.add_patch(patches.FancyBboxPatch((35, 10), 15, 20, boxstyle="round,pad=1", facecolor='#f3e5f5', edgecolor='#7b1fa2', lw=1.5))
ax.text(42.5, 20, 'FC -> ReLU\n-> FC -> Sigmoid', ha='center', va='center', fontweight='bold')
ax.annotate('', xy=(55, 20), xytext=(51, 20), arrowprops=dict(arrowstyle='->', lw=1.5))

# Weights
ax.add_patch(patches.Rectangle((56, 15), 4, 10, facecolor='#fff3e0', edgecolor='#ef6c00', lw=1.5))
ax.text(58, 12, 'Weights', ha='center', fontsize=9)
ax.annotate('', xy=(65, 20), xytext=(61, 20), arrowprops=dict(arrowstyle='->', lw=1.5))

# Scale (Multiply)
ax.add_patch(patches.Circle((68, 20), 3, facecolor='white', edgecolor='#ef6c00', lw=2))
ax.text(68, 20, 'X', ha='center', va='center', fontsize=12, color='#ef6c00', fontweight='bold')

# Skip connection
ax.plot([9, 9, 68, 68], [10, 5, 5, 17], color='gray', lw=1.5, linestyle='--')
ax.annotate('', xy=(68, 17), xytext=(68, 16), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Output
ax.annotate('', xy=(76, 20), xytext=(72, 20), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.add_patch(patches.Rectangle((77, 10), 8, 20, facecolor='#e3f2fd', edgecolor='#1565c0', lw=1.5))
ax.text(81, 32, 'Recalibrated\nFeature', ha='center')

plt.title('Figure 3: Detailed Structure of Frequency Attention Module (FAM)', fontweight='bold', fontsize=14)
plt.savefig('TKDD_Figures/Fig3_FAM_Details.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图 3 (FAM 细节图) 绘制完成！")
