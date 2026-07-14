import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm

# --- 核心：强行指定 4090 D 显卡，跳过不兼容的 TITAN Xp ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 导入各基线模型定义 ---
# SAFE 基线使用你本地修改过的 resnet.py (ResNet-18)
try:
    from models.resnet import resnet18 as SAFE_Baseline_Net
except ImportError:
    from torchvision.models import resnet18 as SAFE_Baseline_Net

def load_model(name, path):
    print(f"📦 正在装载模型: {name} ...")
    
    if name == 'CNNDet':
        # CNNDet 官方权重 (269MB) 必须配合 ResNet-50 躯壳 [cite: 327]
        model = models.resnet50(num_classes=1).to(DEVICE)
    elif name == 'SAFE':
        # SAFE 基线在论文中定义为 ResNet-18 [cite: 79, 156]
        model = SAFE_Baseline_Net(num_classes=1).to(DEVICE)
    elif name == 'FreqNet':
        from models.freqnet_model import FreqNet # [cite: 324]
        model = FreqNet().to(DEVICE)
    elif name == 'UnivFD':
        # UnivFD 通常基于 ResNet-50 或 CLIP-ViT [cite: 306]
        from models.univfd_model import resnet50 as UnivFD_Net
        model = UnivFD_Net(num_classes=1).to(DEVICE)
    else:
        return None

    # 加载灵魂 (权重)
    if not os.path.exists(path):
        print(f"❌ 错误: 找不到权重文件 {path}")
        return None
        
    sd = torch.load(path, map_location=DEVICE)
    # 处理字典键值对，有的权重包裹在 'model' 或 'state_dict' 下
    state_dict = sd['model'] if 'model' in sd else (sd['state_dict'] if 'state_dict' in sd else sd)
    
    # 强制兼容性加载
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# --- 2. 评测核心函数 (增加 AUC 保护) ---
@torch.no_grad()
def run_eval(model, loader, is_wfir=False):
    probs, labels = [], []
    for imgs, lbls in tqdm(loader, desc="Running Inference", leave=False):
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        # 适配输出维度：多分类取 index 1，单分类用 sigmoid
        p = torch.softmax(out, dim=1)[:, 1] if out.shape[1] > 1 else torch.sigmoid(out).squeeze()
        
        # 处理单样本 batch 的情况
        if p.dim() == 0:
            probs.append(p.item())
        else:
            probs.extend(p.cpu().numpy())
        labels.extend(lbls.numpy())
    
    preds = [1 if p > 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds) * 100
    
    # 保护逻辑：如果测试集只有一个类别 (如全是 Fake)，AUC 设为 0
    try:
        auc = roc_auc_score(labels, probs) * 100
    except ValueError:
        auc = 0.0 
    
    res = {'ACC': acc, 'AUC': auc}
    if is_wfir: # 对应 Table 3 指标 [cite: 202, 203]
        res['Prec'] = precision_score(labels, preds, zero_division=0) * 100
        res['Rec'] = recall_score(labels, preds, zero_division=0) * 100
    return res

# --- 3. 运行配置 ---
if __name__ == "__main__":
    print(f"🚀 已成功启动！当前显卡: {torch.cuda.get_device_name(0)}")
    
    # 论文测试集路径 [cite: 177, 178]
    test_sets = {
        'Deepfake': './temp_eval_env/deepfake',
        'Guided_Diff': './temp_eval_env/diffusion/guided',
        'WFIR': './temp_eval_env/wfir'
    }

    # 基线模型与权重对应关系
    baselines = {
        'CNNDet': './baselines_weights/cnndet_blur_jpg_prob0.1.pth',
        'SAFE': './baselines_weights/checkpoint-best.pth'
    }

    print("\n" + "="*50 + "\n开始收集 Table 1 & Table 3 实验数据\n" + "="*50)

    for m_name, w_path in baselines.items():
        model = load_model(m_name, w_path)
        if model is None: continue
        
        print(f"\n📊 模型评测结果: [{m_name}]")
        for d_name, d_path in test_sets.items():
            if not os.path.exists(d_path): continue
            
            # 统一预处理 [cite: 51]
            dataset = datasets.ImageFolder(d_path, transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
            loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
            
            res = run_eval(model, loader, is_wfir=(d_name=='WFIR'))
            
            output = f"  📍 {d_name:12} | ACC: {res['ACC']:.2f}% | AUC: {res['AUC']:.2f}%"
            if 'Prec' in res:
                output += f" | Precision: {res['Prec']:.2f}% | Recall: {res['Rec']:.2f}%"
            print(output)