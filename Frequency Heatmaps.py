import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms

def get_average_spectrum(image_folder, limit=50):
    """计算文件夹中图像的平均频域对数幅度谱"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    fft_accum = None
    count = 0
    
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: return None

    for i, file in enumerate(files[:limit]):
        img_path = os.path.join(image_folder, file)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img) # [1, 256, 256]
            
            # FFT 变换
            fft = torch.fft.fft2(img_tensor)
            fft_shift = torch.fft.fftshift(fft)
            magnitude = torch.log(torch.abs(fft_shift) + 1e-6)
            
            if fft_accum is None:
                fft_accum = magnitude
            else:
                fft_accum += magnitude
            count += 1
        except:
            continue
            
    if count == 0: return None
    return (fft_accum / count).squeeze().numpy()

# === 配置路径 (请修改为您服务器上的真实路径) ===
# 建议分别选取：真实人脸、ProGAN(GAN代表)、StableDiffusion(扩散代表)
path_real = "datasets/test1_ForenSynths/test/progan/0_real" 
path_gan = "datasets/test1_ForenSynths/test/progan/1_fake"
path_diff = "datasets/test3_Ojha/test/guided/1_fake"

# 计算频谱
avg_real = get_average_spectrum(path_real)
avg_gan = get_average_spectrum(path_gan)
avg_diff = get_average_spectrum(path_diff)

# 绘图
if avg_real is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 统一 Scale 以便对比
    vmin = min(avg_real.min(), avg_gan.min(), avg_diff.min())
    vmax = max(avg_real.max(), avg_gan.max(), avg_diff.max())

    im0 = axes[0].imshow(avg_real, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title("Real Images (Average Spectrum)")
    axes[0].axis('off')
    
    im1 = axes[1].imshow(avg_gan, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title("GAN (ProGAN) Artifacts")
    axes[1].axis('off')
    
    im2 = axes[2].imshow(avg_diff, cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title("Diffusion (Guided) Artifacts")
    axes[2].axis('off')

    # 添加 Colorbar
    fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    plt.savefig('Figure_Spectral_Analysis.pdf', dpi=300)
    print("Frequency Heatmap generated.")
else:
    print("Path error. Please check dataset paths.")