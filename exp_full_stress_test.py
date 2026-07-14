import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.safe_mb import resnet18 
import os
import warnings
import sys

# 忽略警告
warnings.filterwarnings("ignore")

# --- 核心配置 ---
CHECKPOINT = "results/ablation/ours_attention/checkpoint-best.pth"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 根据你的目录结构定义的测试集路径
# 如果报错找不到文件，请检查这里的路径是否正确
DATASETS = {
    "Deepfake (Target OOD)": "datasets/test1_ForenSynths/test/deepfake",
    "Diffusion (Ojha Guided)": "datasets/test3_Ojha/test/guided", 
    "ProGAN (In-Domain)": "datasets/test1_ForenSynths/test/progan"
}

# --- 噪声注入类 ---
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.std == 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# --- 数据加载 ---
def get_dataloader(root_dir, noise_level=0.0):
    if not os.path.exists(root_dir):
        print(f"⚠️  Path not found: {root_dir}")
        return None
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(0., noise_level)
    ])
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- 评估函数 ---
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    limit = 50 # 为了速度，每组只跑前50个Batch (约3200张图)，足够看趋势了
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= limit: break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0: return 0.0
    return 100 * correct / total

# --- Hook: 强制权重为1 ---
def hook_force_ones(module, input, output):
    return torch.ones_like(output)

# --- 主程序 ---
if __name__ == "__main__":
    print(f"🚀 Starting Full Stress Test on GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载模型
    model = resnet18(pretrained=False, ablation_mode='ours')
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model_dict = model.state_dict()
    # 过滤不匹配的键
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    # 2. 遍历每个数据集
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*20} Testing Set: {ds_name} {'='*20}")
        
        # 检查路径
        test_loader = get_dataloader(ds_path, 0.0)
        if test_loader is None: continue
        
        print(f"{'Noise (σ)':<12} | {'Baseline (Ours)':<18} | {'Naive (Force 1.0)':<18} | {'Gap (Naive-Base)':<18}")
        print("-" * 75)

        # 3. 遍历噪声等级
        for noise in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]:
            loader = get_dataloader(ds_path, noise)
            
            # A. 测 Baseline (含 FAM)
            acc_base = evaluate(model, loader)
            
            # B. 测 Naive (Hook 强制为 1.0)
            hooks = []
            for m in model.modules():
                if isinstance(m, nn.Sigmoid):
                    hooks.append(m.register_forward_hook(hook_force_ones))
            
            acc_naive = evaluate(model, loader)
            
            # 移除 Hook
            for h in hooks: h.remove()
            
            # 打印行
            gap = acc_naive - acc_base
            print(f"{noise:<12.2f} | {acc_base:<18.2f} | {acc_naive:<18.2f} | {gap:+.2f}")
            
    print("\n✅ All tests finished.")
