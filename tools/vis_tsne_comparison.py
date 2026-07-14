import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import models.safe_mb
from data.datasets import TrainDataset
from utils import remap_checkpoint_keys 

# ================= 配置区域 =================
DATA_PATH = 'datasets/test3_Ojha/test' 
BASELINE_CKPT = 'results/ablation/baseline_hh_only_SAFE_bak/checkpoint-best.pth' 
OURS_CKPT = 'results/ablation/ours_attention/checkpoint-best.pth' 
OUTPUT_DIR = 'figures/vis_tsne'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 特征提取器 =================
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.model.fc = torch.nn.Identity() 

    def forward(self, x):
        return self.model(x)

def load_features_and_labels(extractor, dataloader, device, desc):
    extractor.eval()
    features = []
    labels = []
    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc=desc):
            samples = samples.to(device)
            feat = extractor(samples)
            features.append(feat.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# ================= 学术级绘图模块 =================
def plot_tsne(features, labels, title, filename):
    print(f"Starting t-SNE reduction for {title}...")
    # ✅ 修复版本冲突：移除了不兼容的 n_iter 和 n_jobs 参数
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # ✅ 学术规范排版
    plt.figure(figsize=(8, 6))
    
    # 学术经典对比色：深蓝 (Real) vs 砖红 (Fake)
    colors = ['#1f77b4', '#d62728'] 
    markers = ['o', '^']
    classes = ['Real Images', 'Generated Images']
    
    for i in range(2):
        indices = np.where(labels == i)
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], 
                    c=colors[i], marker=markers[i], label=classes[i], 
                    s=25, alpha=0.7, edgecolors='none')
    
    # 标签与字体规范化
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=13)
    plt.ylabel('t-SNE Dimension 2', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # 去除右侧和顶部边框 (顶刊常用风格)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(fontsize=12, loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()
    
    # 高清输出，防止裁切
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE Figure saved: {filename}")

# ================= 主作战流程 =================
if __name__ == '__main__':
    class DummyArgs:
        input_size = 256
        blur_sigma = None
        jpeg_factor = None
        ablation_mode = 'ours'
        transform_mode = 'crop'
        mask_ratio = None
        mask_patch_size = None
        eval_data_path = DATA_PATH

    args = DummyArgs()
    print("Initializing dataset...")
    dataset_val = TrainDataset(is_train=False, args=args)
    
    dataset_size = len(dataset_val)
    sample_size = min(4000, dataset_size)
    print(f"Dataset total size: {dataset_size}, sampling {sample_size} for t-SNE...")
    
    torch.manual_seed(42) 
    indices = torch.randperm(dataset_size)[:sample_size].tolist()
    dataset_val_subset = Subset(dataset_val, indices)
    
    dataloader = DataLoader(dataset_val_subset, batch_size=64, shuffle=False, num_workers=8)

    # 1️⃣ 处理 Baseline 模型
    print("\n[Step 1/2] Processing Baseline (SAFE)...")
    model_baseline = models.safe_mb.resnet18(num_classes=2, ablation_mode='baseline')
    
    if os.path.exists(BASELINE_CKPT):
        checkpoint = torch.load(BASELINE_CKPT, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            mapped_state_dict = remap_checkpoint_keys(checkpoint['model'])
            model_baseline.load_state_dict(mapped_state_dict, strict=False)
        else:
            model_baseline.load_state_dict(checkpoint, strict=False)
    else:
        print(f"⚠️ Warning: 找不到 Baseline 权重 {BASELINE_CKPT}！")
        
    model_baseline.to(device)
    extractor_baseline = FeatureExtractor(model_baseline)
    feats_base, labels_base = load_features_and_labels(extractor_baseline, dataloader, device, "Extracting Baseline Feats")
    
    plot_tsne(feats_base, labels_base, "SAFE (Baseline): Feature Space on Diffusion", os.path.join(OUTPUT_DIR, 'tsne_baseline_diffusion.png'))

    # 2️⃣ 处理 Ours 模型
    print("\n[Step 2/2] Processing Ours (AMFF-Net)...")
    model_ours = models.safe_mb.resnet18(num_classes=2, ablation_mode='ours')
    
    if os.path.exists(OURS_CKPT):
        checkpoint = torch.load(OURS_CKPT, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            mapped_state_dict = remap_checkpoint_keys(checkpoint['model'])
            model_ours.load_state_dict(mapped_state_dict, strict=False)
        else:
            model_ours.load_state_dict(checkpoint, strict=False)
    else:
        print(f"⚠️ Warning: 找不到 Ours 权重 {OURS_CKPT}！")
        
    model_ours.to(device)
    extractor_ours = FeatureExtractor(model_ours)
    feats_ours, labels_ours = load_features_and_labels(extractor_ours, dataloader, device, "Extracting Ours Feats")
    
    plot_tsne(feats_ours, labels_ours, "AMFF-Net (Ours): Feature Space on Diffusion", os.path.join(OUTPUT_DIR, 'tsne_ours_diffusion.png'))

    print("\n🎉 Congratulations! Figure 6 (t-SNE Comparison) is READY!")