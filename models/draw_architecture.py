import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_amff_net_architecture(save_path='figures/architecture_diag.png'):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 70)
    ax.axis('off') # 隐藏坐标轴
    
    # 统一设置
    fontsize_title = 14
    fontsize_label = 11
    
    # 学术蓝 (主色调) & 学术橙 (创新点)
    color_blue = '#e3f2fd'
    color_blue_stroke = '#1565c0'
    color_orange = '#fff3e0'
    color_orange_stroke = '#ef6c00'

    # --- 1. Input & DWT ---
    # Input Image
    ax.add_patch(patches.Rectangle((5, 25), 15, 20, fill=True, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(12.5, 35, 'Input\nImage', ha='center', va='center', fontsize=fontsize_label)
    
    # Arrow to DWT
    ax.arrow(20, 35, 8, 0, head_width=2, head_length=3, fc='black', ec='black')
    
    # DWT Module
    ax.add_patch(patches.FancyBboxPatch((28, 25), 15, 20, boxstyle="round,pad=1", facecolor=color_blue, edgecolor=color_blue_stroke, linewidth=1.5))
    ax.text(35.5, 35, '2D-DWT', ha='center', va='center', fontsize=fontsize_label, fontweight='bold')

    # --- 2. Band Decomposition (Vertical Layout) ---
    band_names = ['LL', 'LH', 'HL', 'HH']
    band_y_pos = [55, 45, 35, 25]
    for name, y in zip(band_names, band_y_pos):
        # Arrows from DWT
        ax.arrow(43, 35, 5, y-35, head_width=1.5, head_length=2.5, fc='gray', ec='gray', linewidth=0.5)
        # Band rectangles
        ax.add_patch(patches.Rectangle((48, y-5), 10, 10, fill=True, facecolor='white', edgecolor='black', linewidth=1))
        ax.text(53, y, name, ha='center', va='center', fontsize=fontsize_label-1)
        # Arrows to Concat
        ax.arrow(58, y, 5, 35-y, head_width=1.5, head_length=2.5, fc='gray', ec='gray', linewidth=0.5)

    # --- 3. Concat & Concat Tensor ---
    # Concat Node
    ax.text(65, 35, 'Concat', ha='center', va='center', fontsize=fontsize_label-1, color='gray')
    ax.arrow(68, 35, 5, 0, head_width=2, head_length=3, fc='black', ec='black')
    
    # Concat Tensor (Long rectangle)
    ax.add_patch(patches.Rectangle((73, 15), 5, 40, fill=True, facecolor=color_blue, edgecolor=color_blue_stroke, linewidth=1.5))
    ax.text(75.5, 35, 'C x H/2 x W/2\nFeature Map', ha='center', va='center', rotation=90, fontsize=fontsize_label-1)

    # --- 4. INNOVATION: FAM (The highlighted part) ---
    # Arrow to FAM
    ax.arrow(78, 35, 5, 0, head_width=2, head_length=3, fc='black', ec='black')
    
    # FAM Container (虚线框突出创新)
    ax.add_patch(patches.FancyBboxPatch((83, 10), 25, 50, boxstyle="round,pad=1", facecolor=color_orange, edgecolor=color_orange_stroke, linewidth=2, linestyle='--'))
    ax.text(95.5, 55, 'Frequency Attention\nModule (FAM)', ha='center', va='center', fontsize=fontsize_label, fontweight='bold', color=color_orange_stroke)
    
    # FAM Internal logic (GAP & FCs)
    ax.add_patch(patches.Circle((90, 35), 3, fill=True, facecolor='white', edgecolor='black'))
    ax.text(90, 35, 'GAP', ha='center', va='center', fontsize=fontsize_label-2)
    ax.arrow(93, 35, 3, 0, head_width=1, head_length=2, fc='black', ec='black')
    
    ax.add_patch(patches.Rectangle((96, 25), 5, 20, fill=True, facecolor='white', edgecolor='black'))
    ax.text(98.5, 35, 'FCs\nReLU', ha='center', va='center', fontsize=fontsize_label-2)
    ax.arrow(101, 35, 3, 0, head_width=1, head_length=2, fc='black', ec='black')
    
    ax.add_patch(patches.Rectangle((104, 25), 4, 20, fill=True, facecolor='white', edgecolor=color_orange_stroke))
    ax.text(106, 35, 'Sigmoid\nWeights', ha='center', va='center', rotation=90, fontsize=fontsize_label-3, color=color_orange_stroke)
    
    # Excitation Arrow
    ax.arrow(108, 35, 3, 0, head_width=1.5, head_length=2, fc='black', ec='black')
    
    # Multiplication Node (⊕)
    ax.add_patch(patches.Circle((113, 35), 3, fill=True, facecolor='white', edgecolor=color_orange_stroke, linewidth=1.5))
    ax.text(113, 35, r'$\otimes$', ha='center', va='center', fontsize=fontsize_title, color=color_orange_stroke)

    # --- 5. Backbone & Output ---
    # Arrow from FAM to Backbone
    ax.arrow(116, 35, 5, 0, head_width=2, head_length=3, fc='black', ec='black')
    
    # Backbone (ResNet-18) - 梯形
    backbone_poly = patches.Polygon([[121, 20], [130, 25], [130, 45], [121, 50]], fill=True, facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=1.5)
    ax.add_patch(backbone_poly)
    ax.text(125.5, 35, 'ResNet-18', ha='center', va='center', fontsize=fontsize_label, rotation=90)
    
    # Final Output Arrow
    ax.arrow(130, 35, 5, 0, head_width=2, head_length=3, fc='black', ec='black')
    ax.text(138, 35, 'Real\n/\nFake', ha='center', va='center', fontsize=fontsize_label, fontweight='bold', color='#c62828')

    plt.title('Figure 2: Overall Architecture of AMFF-Net', fontsize=fontsize_title, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to: {save_path}")

if __name__ == '__main__':
    # 确保 figures 目录存在
    os.makedirs('figures', exist_ok=True)
    draw_amff_net_architecture()