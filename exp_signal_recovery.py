import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.safe_mb import resnet18 
import os
import warnings

# 忽略显卡兼容性警告
warnings.filterwarnings("ignore")

# --- 配置 ---
CHECKPOINT = "results/ablation/ours_attention/checkpoint-best.pth"
DATA_DIR = "temp_eval_env/deepfake" 
BATCH_SIZE = 64 # 4090D 显存很大，可以开大一点

# 强制使用 CUDA (稍后我们在命令行指定只看 GPU 0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloader():
    """加载 Deepfake 测试数据"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def evaluate(model, loader, description):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    # 只需要跑一部分数据就能看出趋势
    limit_batches = 50 
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= limit_batches: break
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0: return 0.0
    acc = 100 * correct / total
    print(f"[{description}] Accuracy: {acc:.2f}% (Samples: {total})")
    return acc

# --- Hook 函数定义 ---
def hook_force_ones(module, input, output):
    """Exp 1: 强制权重为 1.0 (模拟 Naive MB)"""
    return torch.ones_like(output)

def hook_signal_boost(module, input, output):
    """Exp 2: 信号唤醒 (权重 x 2.0)"""
    return output * 2.0

# --- 主程序 ---
if __name__ == "__main__":
    print(f"Using Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    
    print(f"Loading checkpoint from: {CHECKPOINT}")
    
    # 1. 加载模型
    model = resnet18(pretrained=False, ablation_mode='ours')
    
    # --- 修复点：添加 weights_only=False 以加载旧版 Checkpoint ---
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # 过滤参数
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(state_dict, strict=False)
    
    model.to(DEVICE)
    
    loader = get_dataloader()
    print(f"Testing on Deepfake dataset...")
    print("-" * 60)

    # 阶段 1: Baseline
    acc_base = evaluate(model, loader, "Baseline (Weights ≈ 0.5)")

    # 阶段 2: Exp 1 - 强制直通
    hooks = []
    sigmoid_found = False
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Sigmoid):
            h = module.register_forward_hook(hook_force_ones)
            hooks.append(h)
            sigmoid_found = True
    
    if not sigmoid_found:
        print("Error: Could not find Sigmoid layer to hook!")
    else:
        acc_exp1 = evaluate(model, loader, "Exp 1: Force Weights = 1.0 (Naive Sim)")
        
    for h in hooks: h.remove()
    hooks = []

    # 阶段 3: Exp 2 - 信号唤醒
    for name, module in model.named_modules():
        if isinstance(module, nn.Sigmoid):
            h = module.register_forward_hook(hook_signal_boost)
            hooks.append(h)
            
    acc_exp2 = evaluate(model, loader, "Exp 2: Signal Boost (Weights * 2.0)")
    
    print("-" * 60)
    print("ANALYSIS REPORT:")
    if acc_exp1 > acc_base + 10:
        print("✅ HYPOTHESIS CONFIRMED: Signal Dampening is the culprit!")
        print(f"   - Original Acc: {acc_base:.2f}% (Suffering from weight suppression)")
        print(f"   - Unsuppressed Acc: {acc_exp1:.2f}% (Features are actually valid!)")
    else:
        print("❓ Hypothesis Inconclusive.")
