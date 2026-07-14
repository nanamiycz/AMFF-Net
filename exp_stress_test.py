import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.safe_mb import resnet18 
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# --- 配置 ---
CHECKPOINT = "results/ablation/ours_attention/checkpoint-best.pth"
DATA_DIR = "temp_eval_env/deepfake" 
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AddGaussianNoise(object):
    """添加高斯噪声来模拟困难样本"""
    def __init__(self, mean=0., std=0.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.std == 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_dataloader(noise_level=0.0):
    """加载数据并注入指定强度的噪声"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(0., noise_level) # <--- 关键：人为制造困难
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- Hook ---
def hook_force_ones(module, input, output):
    return torch.ones_like(output) # 模拟无注意力 (Naive)

if __name__ == "__main__":
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = resnet18(pretrained=False, ablation_mode='ours')
    
    # 加载权重
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    print(f"\n{'='*20} STRESS TEST START {'='*20}")
    print(f"{'Noise Level':<15} | {'Baseline (0.5x)':<15} | {'Exp1 (Naive 1.0x)':<15} | {'Gap':<10}")
    print("-" * 65)

    # 测试不同的噪声强度
    for noise in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]:
        loader = get_dataloader(noise_level=noise)
        
        # 1. 测 Baseline
        acc_base = evaluate(model, loader)
        
        # 2. 测 Naive (Exp 1)
        hooks = []
        for m in model.modules():
            if isinstance(m, nn.Sigmoid):
                hooks.append(m.register_forward_hook(hook_force_ones))
        
        acc_naive = evaluate(model, loader)
        
        # 清理 Hook
        for h in hooks: h.remove()
        
        print(f"{noise:<15.2f} | {acc_base:<15.2f} | {acc_naive:<15.2f} | {acc_naive - acc_base:+.2f}")

    print(f"{'='*56}")
    print("Interpretation:")
    print("If 'Gap' increases as 'Noise Level' goes up, your hypothesis is proven:")
    print("Signal dampening (0.5x) makes the model fragile to noise!")
