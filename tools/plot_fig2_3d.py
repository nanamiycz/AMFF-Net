import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs('TKDD_Figures', exist_ok=True)

# 🚀 核心魔法：手搓 3D 立体块绘制函数
def draw_cuboid(ax, x, y, w, h, d, c_front, c_side, c_top, alpha=1.0, zorder=2):
    # 正面 (Front)
    front = patches.Rectangle((x, y), w, h, facecolor=c_front, edgecolor='black', lw=1, alpha=alpha, zorder=zorder+1)
    ax.add_patch(front)
    # 顶部 (Top) - 使用多边形计算透视
    top = patches.Polygon([[x, y+h], [x+d, y+h+d], [x+w+d, y+h+d], [x+w, y+h]], facecolor=c_top, edgecolor='black', lw=1, alpha=alpha, zorder=zorder)
    ax.add_patch(top)
    # 侧面 (Side) - 使用多边形计算透视
    side = patches.Polygon([[x+w, y], [x+w+d, y+d], [x+w+d, y+h+d], [x+w, y+h]], facecolor=c_side, edgecolor='black', lw=1, alpha=alpha, zorder=zorder)
    ax.add_patch(side)

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 165)
ax.set_ylim(0, 80)
ax.axis('off')

# 🎨 顶刊 3D 配色方案
blue_f, blue_s, blue_t = '#64b5f6', '#42a5f5', '#2196f3' # 蓝色特征块
grey_f, grey_s, grey_t = '#f5f5f5', '#eeeeee', '#e0e0e0' # 被丢弃的灰色块
org_f, org_s, org_t = '#ffb74d', '#ffa726', '#ff9800'    # 注意力激活的橙色块

# ================= 1. Input Image =================
ax.add_patch(patches.Rectangle((5, 30), 10, 15, facecolor='white', edgecolor='black', lw=1.5))
ax.text(10, 37.5, 'Input\n$X$', ha='center', va='center', fontsize=12)
ax.annotate('', xy=(22, 37.5), xytext=(16, 37.5), arrowprops=dict(arrowstyle='->', lw=1.5))

# ================= 2. DWT =================
ax.add_patch(patches.FancyBboxPatch((23, 27.5), 10, 20, boxstyle="round,pad=1", facecolor='#e3f2fd', edgecolor='#1565c0', lw=1.5))
ax.text(28, 37.5, 'DWT', ha='center', va='center', fontsize=12, fontweight='bold')

# ================= 3. 四大频段 (3D 显示 + LL 红叉) =================
bands = ['HH (High)', 'HL (Mid)', 'LH (Mid)', 'LL (Low)']
y_pos = [15, 30, 45, 60]

for name, y in zip(bands, y_pos):
    # DWT 引出的分支箭头
    ax.plot([34, 40, 40, 43], [37.5, 37.5, y+4, y+4], color='black', lw=1)
    ax.annotate('', xy=(45, y+4), xytext=(43, y+4), arrowprops=dict(arrowstyle='->', lw=1))
    
    if 'LL' in name:
        # LL 频段：画成半透明灰色 3D 块
        draw_cuboid(ax, 46, y, 8, 8, 4, grey_f, grey_s, grey_t, alpha=0.5)
        ax.text(50, y-4, name, ha='center', va='center', fontsize=11, color='gray')
        # ❌ 手搓绝杀红叉
        ax.plot([43, 59], [y-2, y+14], color='red', lw=3, zorder=5)
        ax.plot([43, 59], [y+14, y-2], color='red', lw=3, zorder=5)
        ax.text(51, y+18, 'Discarded\n(Semantic Bias)', ha='center', color='red', fontsize=10, fontweight='bold')
    else:
        # 中高频段：画成饱满的蓝色 3D 块
        draw_cuboid(ax, 46, y, 8, 8, 4, blue_f, blue_s, blue_t)
        ax.text(50, y-4, name, ha='center', va='center', fontsize=11)
        ax.annotate('', xy=(66, y+4), xytext=(54+4, y+4), arrowprops=dict(arrowstyle='->', lw=1))

# ================= 4. Concat Tensor (长条 3D 块) =================
draw_cuboid(ax, 67, 15, 5, 38, 5, blue_f, blue_s, blue_t)
ax.text(70, 8, 'Concat Tensor', ha='center', va='center', fontsize=12)
ax.annotate('', xy=(85, 34), xytext=(72+5, 34), arrowprops=dict(arrowstyle='->', lw=1.5))

# ================= 5. FAM 模块 (包含 3D 注意力加权) =================
ax.add_patch(patches.FancyBboxPatch((85, 10), 45, 55, boxstyle="round,pad=2", facecolor='#fff3e0', edgecolor='#ef6c00', lw=2, linestyle='--'))
ax.text(107.5, 68, 'Frequency Attention Module (FAM)', ha='center', va='center', fontsize=13, fontweight='bold', color='#ef6c00')

# 原特征备份 (灰) -> GAP/FC -> 权重 (橙) -> 相乘
draw_cuboid(ax, 88, 20, 3, 28, 3, grey_f, grey_s, grey_t)
ax.annotate('', xy=(97, 34), xytext=(91+3, 34), arrowprops=dict(arrowstyle='->', lw=1))

ax.add_patch(patches.Rectangle((98, 24), 8, 20, facecolor='white', edgecolor='black'))
ax.text(102, 34, 'GAP\n+\nFCs', ha='center', va='center', fontsize=9)
ax.annotate('', xy=(111, 34), xytext=(106, 34), arrowprops=dict(arrowstyle='->', lw=1))

# 3D 权重向量
draw_cuboid(ax, 112, 20, 2, 28, 2, org_f, org_s, org_t)
ax.annotate('', xy=(121, 34), xytext=(114+2, 34), arrowprops=dict(arrowstyle='->', lw=1))

# 乘法操作
ax.add_patch(patches.Circle((123, 34), 2.5, facecolor='white', edgecolor='#ef6c00', lw=2))
ax.text(123, 34, r'$\otimes$', ha='center', va='center', fontsize=14, color='#ef6c00')
ax.plot([89, 89, 123, 123], [20, 15, 15, 31.5], color='gray', linestyle=':', lw=1.5)
ax.annotate('', xy=(123, 31.5), xytext=(123, 30.5), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# ================= 6. 加权后的 Concat Tensor (更厚、变成橙色的 3D 块) =================
ax.annotate('', xy=(135, 34), xytext=(125.5, 34), arrowprops=dict(arrowstyle='->', lw=1.5))
draw_cuboid(ax, 136, 15, 7, 38, 7, org_f, org_s, org_t) # 增大了 w 和 d，代表被施加了权重
ax.text(140, 8, 'Recalibrated\nTensor', ha='center', va='center', fontsize=12, color='#ef6c00')

# ================= 7. Backbone & Output =================
ax.annotate('', xy=(150, 34), xytext=(143+7, 34), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(158, 38, 'Real', ha='center', va='center', fontsize=14, fontweight='bold', color='#2e7d32') # 深绿色
ax.text(158, 30, 'Fake', ha='center', va='center', fontsize=14, fontweight='bold', color='#c62828') # 深红色

plt.title('Figure 2: Overall Architecture of AMFF-Net (3D Tensors)', fontsize=16, fontweight='bold', y=0.93)
plt.tight_layout()
plt.savefig('TKDD_Figures/Fig2_Architecture_3D_HandRubbed.pdf', dpi=300)
plt.close()
print("✅ 纯代码手搓 3D 版图 2 绘制完成！赶快去查看 TKDD_Figures/Fig2_Architecture_3D_HandRubbed.pdf 吧！")
