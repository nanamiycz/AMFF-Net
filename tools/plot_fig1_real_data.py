import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

images_needed = ['real.png', 'gan.png', 'diffusion.png']
for img in images_needed:
    if not os.path.exists(img):
        print(f"⚠️ 警告: 找不到图片 {img}！请确认是否重命名正确。")
        exit()

plt.style.use('seaborn-v0_8-whitegrid')

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    # 进行 2D-DWT 变换
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    hh_energy = np.abs(HH)
    
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))
    return img_rgb, hh_energy

real_rgb, real_hh = process_image('real.png')
gan_rgb, gan_hh = process_image('gan.png')
diff_rgb, diff_hh = process_image('diffusion.png')

# 保持原有的统一刻度计算，确保能量对比公平
vmin, vmax = 0, np.percentile([real_hh, gan_hh, diff_hh], 99) 

os.makedirs('TKDD_Figures', exist_ok=True)

# ==========================================
# 统一设置更大的画布尺寸，让单独的图片撑满屏幕！
# ==========================================
FIG_SIZE = (8, 8) 

# 1. (a) Real Image
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(real_rgb)
ax.set_title('(a) Real Image', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('TKDD_Figures/1_Real_Image.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. (b) GAN Generated
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(gan_rgb)
ax.set_title('(b) GAN Generated', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('TKDD_Figures/2_GAN_Image.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. (c) Diffusion Generated
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(diff_rgb)
ax.set_title('(c) Diffusion Generated', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('TKDD_Figures/3_Diffusion_Image.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Real HH Spectrum
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(real_hh, cmap='magma', vmin=vmin, vmax=vmax)
ax.set_title('Real HH Spectrum\n(Low Energy)', fontsize=16) # 稍微放大了这里的字体以匹配大图
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('TKDD_Figures/4_Real_HH.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. GAN HH Spectrum
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(gan_hh, cmap='magma', vmin=vmin, vmax=vmax)
ax.set_title('GAN HH Spectrum\n(Clear Artifact Peaks)', fontsize=16, color='#d62728')
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('TKDD_Figures/5_GAN_HH.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Diffusion HH Spectrum
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(diff_hh, cmap='magma', vmin=vmin, vmax=vmax)
ax.set_title('Diffusion HH Spectrum\n(Scattered / Suppressed)', fontsize=16, color='#1f77b4')
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('TKDD_Figures/6_Diffusion_HH.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 顶刊级别图 1 已按要求【放大并分别生成】！共 6 张高清大图保存在 TKDD_Figures 文件夹中。")