import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_amff_net_architecture(save_path='figures/architecture_diag.png'):
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
    for name, y in zip(bands, y_offsets):
        ax.annotate('', xy=(55, y), xytext=(45, 40), arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.add_patch(patches.Rectangle((55, y-4), 10, 8, facecolor='white', edgecolor='black'))
        ax.text(60, y, name, ha='center', va='center', fontsize=10)
        ax.annotate('', xy=(75, 40), xytext=(65, y), arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # 4. Concat Tensor
    ax.add_patch(patches.Rectangle((75, 20), 6, 40, facecolor=color_blue, edgecolor=color_blue_stroke, linewidth=1.5))
    ax.text(78, 40, 'Concat Band Features', ha='center', va='center', rotation=90, fontsize=10)
    ax.annotate('', xy=(90, 40), xytext=(81, 40), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 5. INNOVATION: FAM (The Highlighted Part)
    ax.add_patch(patches.FancyBboxPatch((90, 15), 30, 50, boxstyle="round,pad=1", facecolor=color_orange, edgecolor=color_orange_stroke, linewidth=2, linestyle='--'))
    ax.text(105, 58, 'Frequency Attention Module (FAM)', ha='center', va='center', fontsize=12, fontweight='bold', color=color_orange_stroke)
    
    # FAM Internal: GAP -> FCs -> Sigmoid
    ax.add_patch(patches.Circle((98, 40), 3, facecolor='white', edgecolor='black'))
    ax.text(98, 40, 'GAP', ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(108, 40), xytext=(101, 40), arrowprops=dict(arrowstyle='->'))
    ax.add_patch(patches.Rectangle((108, 30), 8, 20, facecolor='white', edgecolor='black'))
    ax.text(112, 40, 'FCs\nReLU', ha='center', va='center', fontsize=9)
    
    # Reweight (Multiplication)
    ax.add_patch(patches.Circle((125, 40), 4, facecolor='white', edgecolor=color_orange_stroke, linewidth=2))
    ax.text(125, 40, r'$\otimes$', ha='center', va='center', fontsize=18, color=color_orange_stroke)
    ax.annotate('', xy=(121, 40), xytext=(116, 40), arrowprops=dict(arrowstyle='->'))

    # 6. Backbone
    ax.annotate('', xy=(135, 40), xytext=(129, 40), arrowprops=dict(arrowstyle='->', lw=1.5))
    backbone = patches.Polygon([[135, 25], [145, 30], [145, 50], [135, 55]], facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=1.5)
    ax.add_patch(backbone)
    ax.text(140, 40, 'ResNet-18\nBackbone', ha='center', va='center', fontsize=10, rotation=90)

    # 7. Final Output
    ax.annotate('', xy=(155, 40), xytext=(145, 40), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(160, 40, 'Real / Fake', ha='center', va='center', fontsize=12, fontweight='bold', color='#c62828')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Diagram saved to: {save_path}")

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    draw_amff_net_architecture()
