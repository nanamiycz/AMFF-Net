import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 继续用卡1

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
from models.safe_mb import resnet18 

# ================= 辅助函数 =================
def find_latest_checkpoint(root_dir="results/SAFE_reproduction"):
    dirs = glob.glob(os.path.join(root_dir, "*SAFE*"))
    if not dirs: return None
    latest_dir = max(dirs, key=os.path.getmtime)
    return os.path.join(latest_dir, "checkpoint-best.pth")

def find_fake_image(keyword, dataset_name):
    # 专门找 1_fake 文件夹下的图片
    # keyword: e.g., "guided" or "deepfake"
    pattern = os.path.join("datasets", dataset_name, "test", keyword, "1_fake", "*.*")
    files = glob.glob(pattern)
    # 过滤常见图片格式
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return images[0] if images else None

# ================= 主逻辑 =================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 使用设备: {device} (卡1)")

    # 1. 加载模型
    ckpt_path = find_latest_checkpoint()
    print(f"📂 权重: {ckpt_path}")
    
    model = resnet18(pretrained=False, num_classes=2).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 寻找两类典型伪造图片
    # Case A: Guided Diffusion (期望激活 LH/HL)
    img_guided = find_fake_image("guided", "test3_Ojha") 
    # Case B: Deepfake (期望激活 HH) - 如果没有下载test1，这个可能为空
    img_deepfake = find_fake_image("deepfake", "test1_ForenSynths")

    target_images = {}
    if img_guided: target_images["Guided Diffusion (New)"] = img_guided
    if img_deepfake: target_images["Deepfake (Legacy)"] = img_deepfake
    
    if not target_images:
        print("❌ 未找到任何 Fake 图片，请检查数据集路径")
        return

    # 3. 推理并捕获权重
    weights_data = {}
    
    # 定义 Hook
    current_weights = []
    def hook_fn(module, input, output):
        current_weights.append(output.detach().cpu().numpy())
    model.freq_attention.fc[3].register_forward_hook(hook_fn)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for name, path in target_images.items():
        print(f"🖼️ 处理图片: {name} -> {path}")
        img = Image.open(path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        current_weights = [] # 清空
        model(img_t)
        weights_data[name] = current_weights[0].flatten()

    # 4. 绘图 (对比图)
    labels = ['LH1','LH2','LH3', 'HL1','HL2','HL3', 'HH1','HH2','HH3']
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制两组柱状图
    for i, (name, w) in enumerate(weights_data.items()):
        offset = width * i - width/2 if len(weights_data) > 1 else 0
        rects = ax.bar(x + offset, w, width, label=name, alpha=0.8)
        
        # 打印数值方便查看
        print(f"📊 {name} 权重: {w}")

    ax.set_ylabel('Attention Weight')
    ax.set_title('Adaptive Frequency Attention: Guided Diffusion vs. Deepfake')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加区域标注
    plt.axvline(x=2.5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)
    plt.text(1, 1.05, 'LH (Mid)', ha='center', fontweight='bold')
    plt.text(4, 1.05, 'HL (Mid)', ha='center', fontweight='bold')
    plt.text(7, 1.05, 'HH (High)', ha='center', fontweight='bold')

    save_name = 'paper_figure_attention.png'
    plt.savefig(save_name)
    print(f"\n✅ 论文配图已保存为: {save_name}")
    print("👉 请查看这张图，对比 Guided 和 Deepfake 的柱子高低是否不同！")

if __name__ == "__main__":
    main()