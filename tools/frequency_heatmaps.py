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
    
    if not os.path.exists(image_folder):
        print(f"⚠️ 找不到路径: {image_folder}")
        return None

    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: 
        print(f"⚠️ 路径下没有图片: {image_folder}")
        return None

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
        except Exception as e:
            continue
            
    if count == 0: return None
    return (fft_accum / count).squeeze().numpy()

# === 使用之前 tree 命令确认存在的绝对路径 ===
path_real = "datasets/test1_ForenSynths/test/deepfake/0_real" 
path_df   = "datasets/test1_ForenSynths/test/deepfake/1_fake"
path_diff = "datasets/test3_Ojha/test/guided/1_fake"

print("正在计算频域热力图，请稍候...")
# 计算频谱
avg_real = get_average_spectrum(path_real)
avg_df = get_average_spectrum(path_df)
avg_diff = get_average_spectrum(path_diff)

# 绘图
if avg_real is not None and avg_df is not None and avg_diff is not None:
    # 设置大字体以适应期刊排版
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 统一 Scale 以便对比
    vmin = min(avg_real.min(), avg_df.min(), avg_diff.min())
    vmax = max(avg_real.max(), avg_df.max(), avg_diff.max())

    im0 = axes[0].imshow(avg_real, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title("(a) Real Images", fontsize=16)
    axes[0].axis('off')
    
    im1 = axes[1].imshow(avg_df, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title("(b) Deepfake (OOD) Artifacts", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(avg_diff, cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title("(c) Diffusion (Guided) Artifacts", fontsize=16, fontweight='bold')
    axes[2].axis('off')

    # 添加 Colorbar
    fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    plt.savefig('Figure_Spectral_Analysis.pdf', dpi=300)
    print("✅ 热力图已成功生成: Figure_Spectral_Analysis.pdf")
else:
    print("❌ 生成失败，请检查路径。")
