import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

sys.path.append(os.path.abspath('.'))
device = torch.device('cuda:0')
print("🚀 启动 5% 数据持续学习 (Continual Learning) 微调...")

# 1. 基础配置
data_path = 'datasets/test2_Self-Synthesis/test'
base_ckpt = 'results/ablation_seeds/ours_seed2026/checkpoint-19.pth'
epochs = 3
batch_size = 32
lr = 1e-5

# 2. 准备 5% 旧数据 (Legacy Budget)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
full_dataset = datasets.ImageFolder(data_path, transform=transform)

# =================================================================
# 【核心修复】：强行纠正 ImageFolder 瞎标的分类！
# 把所有 0,1,2,3... 强制转为二分类：0 (real) 和 1 (fake)
# =================================================================
for i in range(len(full_dataset.samples)):
    path, _ = full_dataset.samples[i]
    label = 0 if 'real' in path.lower() else 1
    full_dataset.samples[i] = (path, label)
    full_dataset.targets[i] = label

num_samples = int(0.05 * len(full_dataset))
np.random.seed(42)
subset_indices = np.random.choice(len(full_dataset), num_samples, replace=False)
train_loader = DataLoader(Subset(full_dataset, subset_indices), batch_size=batch_size, shuffle=True)
print(f"📊 成功载入 5% Legacy 数据: {num_samples} 张图片 (标签已校准).")

def load_model():
    model = None
    try:
        import models.safe_mb as smb
        model = smb.SAFE_MB(smb.BasicBlock, [2, 2, 2, 2], ablation_mode='ours', num_classes=2)
    except Exception:
        try:
            import models.resnet as smb_res
            model = smb_res.SAFE_MB(smb_res.BasicBlock, [2, 2, 2, 2], ablation_mode='ours', num_classes=2)
        except Exception:
            try:
                import models.safe_mb as smb
                model = smb.SAFE_MB(smb.BasicBlock, [2, 2, 2, 2], ablation_mode='ours', nb_classes=2)
            except Exception:
                pass
                
    if model is None:
        raise RuntimeError("模型还是没拉起来！")
        
    state_dict = torch.load(base_ckpt, map_location='cpu', weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

# ==========================================
# 策略 1: Naive Fine-Tuning (天真微调)
# ==========================================
print("\n[1/2] 正在运行: Naive Fine-Tuning ...")
model_naive = load_model()
model_naive.train()
optimizer_naive = torch.optim.AdamW(model_naive.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer_naive.zero_grad()
        loss = F.cross_entropy(model_naive(x), y)
        loss.backward()
        optimizer_naive.step()
        total_loss += loss.item()
    print(f"  > Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

torch.save({'model': model_naive.state_dict()}, 'weight_naive.pth')
print("✅ 已保存: weight_naive.pth")

# ==========================================
# 策略 2: Logits Distillation (知识蒸馏)
# ==========================================
print("\n[2/2] 正在运行: Logits Distillation (Memory-Free) ...")
teacher = load_model()
teacher.eval() # 老师只推理不学习
model_kd = load_model()
model_kd.train() # 学生学习
optimizer_kd = torch.optim.AdamW(model_kd.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer_kd.zero_grad()
        
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = model_kd(x)
        
        loss_ce = F.cross_entropy(s_logits, y) 
        loss_kd_val = F.mse_loss(s_logits, t_logits) 
        
        loss = 0.5 * loss_ce + 0.5 * loss_kd_val
        loss.backward()
        optimizer_kd.step()
        total_loss += loss.item()
    print(f"  > Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

torch.save({'model': model_kd.state_dict()}, 'weight_kd.pth')
print("✅ 已保存: weight_kd.pth")
print("🎉 微调大功告成！")
