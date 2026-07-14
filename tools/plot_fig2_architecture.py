import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs('TKDD_Figures', exist_ok=True)

def draw_amff_net_architecture_updated():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 80)
    ax.axis('off')
    
    color_blue = '#e3f2fd'
    color_blue_stroke = '#1565c0'
    color_orange = '#fff3e0'
    color_orange_stroke = '#ef6c00'

    # 1. Input Image
    ax.add_patch(patches.Rectangle((5, 30), 15, 20, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(12.5, 40, 'Input\nImage', ha='center', va='center', fontsize=12)
    ax.annotate('', xy=(30, 40), xytext=(20, 40), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 2. DWT Module
    ax.add_patch(patches.FancyBboxPatch((30, 30), 15, 20, boxstyle="round,pad=1", facecolor=color_blue, edgecolor=color_blue_stroke, linewidth=1.5))
    ax.text(37.5, 40, '2D-DWT', ha='center', va='center', fontsize=12, fontweight='bold')

    # 3. Frequency Bands (LL, LH, HL, HH)
    bands = ['LL', 'LH', 'HL', 'HH']
    y_offsets = [60, 47, 33, 20]
    
    for i, (name, y) in enumerate(zip(bands, y_offsets)):
        # DWT -> Bands
        ax.annotate('', xy=(55, y), xytext=(45, 40), arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        # 视觉弱化 LL 频段
        face_color = '#f5f5f5' if name == 'LL' else 'white'
        edge_color = 'lightgray' if name == 'LL' else 'black'
        text_color = 'gray' if name == 'LL' else 'black'
        
        ax.add_patch(patches.Rectangle((55, y-4), 10, 8, facecolor=face_color, edgecolor=edge_color))
        ax.text(60, y, name, ha='center', va='center', fontsize=10, color=text_color)
        
        if name == 'LL':
            # ⛔ 画上学术红叉，阻断输入 Concat
            ax.plot([67, 73], [y-2, y+2], color='red', lw=2)
            ax.plot([67, 73], [y+2, y-2], color='red', lw=2)
            ax.text(70, y+5, 'Discarded\n(Semantic Int.)', ha='center', va='bottom', fontsize=8, color='red')
            # 灰色虚线表示原本的路径
            ax.plot([65, 75], [y, y], color='lightgray', linestyle='--', lw=1)
        else:
            # Bands -> Concat
            ax.annotate('', xy=(75, y), xytext=(65, y), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 4. Concat Tensor (现在只接收3个频段)
    ax.add_patch(patches.Rectangle((75, 16), 6, 36, facecolor=color_blue, edgecolor=color_blue_stroke, linewidth=1.5))
    ax.text(78, 34, 'Concat (LH+HL+HH)', ha='center', va='center', rotation=90, fontsize=10)
    ax.annotate('', xy=(90, 34), xytext=(81, 34), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 5. FAM Module
    ax.add_patch(patches.FancyBboxPatch((90, 15), 30, 40, boxstyle="round,pad=1", facecolor=color_orange, edgecolor=color_orange_stroke, linewidth=2, linestyle='--'))
    ax.text(105, 50, 'Frequency Attention Module (FAM)', ha='center', va='center', fontsize=11, fontweight='bold', color=color_orange_stroke)
    
    # FAM 内核缩略
    ax.add_patch(patches.Circle((96, 34), 3, facecolor='white', edgecolor='black'))
    ax.text(96, 34, 'GAP', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(106, 34), xytext=(99, 34), arrowprops=dict(arrowstyle='->'))
    
    ax.add_patch(patches.Rectangle((106, 26), 8, 16, facecolor='white', edgecolor='black'))
    ax.text(110, 34, 'FCs', ha='center', va='center', fontsize=8)
    
    ax.add_patch(patches.Circle((125, 34), 4, facecolor='white', edgecolor=color_orange_stroke, linewidth=2))
    ax.text(125, 34, r'$\otimes$', ha='center', va='center', fontsize=18, color=color_orange_stroke)
    ax.annotate('', xy=(121, 34), xytext=(114, 34), arrowprops=dict(arrowstyle='->'))

    # 6. Backbone
    ax.annotate('', xy=(135, 34), xytext=(129, 34), arrowprops=dict(arrowstyle='->', lw=1.5))
    backbone = patches.Polygon([[135, 20], [145, 25], [145, 43], [135, 48]], facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=1.5)
    ax.add_patch(backbone)
    ax.text(140, 34, 'ResNet-18', ha='center', va='center', fontsize=10, rotation=90)

    # 7. Final Output
    ax.annotate('', xy=(155, 34), xytext=(145, 34), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(160, 34, 'Real / Fake', ha='center', va='center', fontsize=12, fontweight='bold', color='#c62828')

    plt.tight_layout()
    plt.savefig('TKDD_Figures/Fig2_Architecture_Updated.pdf', dpi=300, bbox_inches='tight')
    print("✅ 图 2 (更新后的整体架构图) 绘制完成！")

if __name__ == '__main__':
    draw_amff_net_architecture_updated()
