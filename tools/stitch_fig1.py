import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 1. 直接指定 TKDD_Figures 文件夹下的精确路径
base_dir = 'TKDD_Figures'

images_top = [
    os.path.join(base_dir, '1_Real_Image.png'),
    os.path.join(base_dir, '2_GAN_Image.png'),
    os.path.join(base_dir, '3_Diffusion_Image.png')
]

images_bottom = [
    os.path.join(base_dir, '4_Real_HH_1.png'),
    os.path.join(base_dir, '5_GAN_HH_1.png'),
    os.path.join(base_dir, '6_Diffusion_HH_1.png')
]

# 检查文件是否都存在
all_images = images_top + images_bottom
for img in all_images:
    if not os.path.exists(img):
        print(f"⚠️ 致命错误: 找不到图片 {img}！请检查目录。")
        exit()

# 2. 创建 2行3列 的大画布 (背景设为白色)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')

# ================= 上排：拼入原图 =================
titles_top = ['(a) Real Image', '(b) GAN Generated', '(c) Diffusion Generated']
for i in range(3):
    img = mpimg.imread(images_top[i])
    axes[0, i].imshow(img)
    axes[0, i].set_title(titles_top[i], fontsize=16, fontweight='bold', pad=15)
    axes[0, i].axis('off') # 关掉坐标轴

# ================= 下排：拼入你的频谱图 =================
# 你底部的图已经自带了颜色条和标题，直接贴上去展示！
for i in range(3):
    img = mpimg.imread(images_bottom[i])
    axes[1, i].imshow(img)
    axes[1, i].axis('off') # 关掉坐标轴

# 自动调整间距
plt.tight_layout()

# 3. 保存拼接后的终极六宫格大图
save_png = os.path.join(base_dir, 'Fig1_Motivation_Spectrum_Stitched.png')
save_pdf = os.path.join(base_dir, 'Fig1_Motivation_Spectrum_Stitched.pdf')

plt.savefig(save_png, dpi=300, bbox_inches='tight')
plt.savefig(save_pdf, bbox_inches='tight')
plt.close()

print(f"✅ 物理拼接完成！已成功读取 TKDD_Figures 下的 6 张图，并拼装为:\n - {save_png}\n - {save_pdf}")