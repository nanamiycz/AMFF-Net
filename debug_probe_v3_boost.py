import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import glob

# ==========================================
# 1. 模型定义 (带“上帝之手”修改版)
# ==========================================
class AdaptiveFrequencyAttention(nn.Module):
    def __init__(self, channel=9, reduction=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 保持结构不变，以便加载权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        # ============================================================
        # 💉 【手术台】 拆解 Sequential，植入兴奋剂
        # ============================================================
        # 原流程：y = self.fc(y)
        # 现流程：手动一步步走，中间插一脚
        
        # 1. 第一层 Linear
        y = self.fc[0](y)
        # 2. ReLU
        y = self.fc[1](y)
        # 3. 第二层 Linear (此时得到 Logits，还没过 Sigmoid)
        logits = self.fc[2](y)
        
        # ------------------------------------------------------------
        # 🔥 GOD MODE: 强制激活高频 (HH)
        # ------------------------------------------------------------
        # 索引 0-2: LH (低频)
        # 索引 3-5: HL (中频)
        # 索引 6-8: HH (高频) <--- 我们要搞它
        
        # 给 HH 通道加上 5.0 的强偏置
        # 这会让 Sigmoid 输出接近 0.99 (强制关注)
        if not self.training: # 只在测试时搞
            logits[:, 6:9] += 5.0 
            # 可选：如果你想更极端，可以抑制低频
            # logits[:, 0:3] -= 2.0 
            
        # 4. Sigmoid 激活
        y = self.fc[3](logits).view(b, c, 1, 1)
        # ------------------------------------------------------------
        
        # [PROBE] 打印现在的权重，看看是不是变成 0.99 了
        if not self.training:
            w = y.mean(dim=0).view(-1).detach().cpu().numpy()
            print(f"   💉 [Boosted Probe] LH: {w[0:3].mean():.4f} | HL: {w[3:6].mean():.4f} | HH: {w[6:9].mean():.4f}")
            
        return x * y.expand_as(x)

class SAFE_MB_Infer(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_attention = AdaptiveFrequencyAttention(channel=9, reduction=3)
        self.freq_fusion = nn.Conv2d(9, 3, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        return nn.Sequential(
            BasicBlock(inplanes, planes, stride),
            BasicBlock(planes, planes)
        )

    def _preprocess_dwt(self, x):
        if not hasattr(self, 'dwt_fn'):
            self.dwt_fn = DWTForward(J=1, mode='symmetric', wave='bior1.3').to(x.device)
        Yl, Yh = self.dwt_fn(x)
        lh, hl, hh = Yh[0][:, :, 0, :, :], Yh[0][:, :, 1, :, :], Yh[0][:, :, 2, :, :]
        x_freq = torch.cat([lh, hl, hh], dim=1)
        return F.interpolate(x_freq, size=(x.shape[-2]//2, x.shape[-1]//2), mode='bilinear')

    def forward(self, x):
        x_freq = self._preprocess_dwt(x)
        x_freq = self.freq_attention(x_freq)
        x = self.freq_fusion(x_freq)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def run_test(img_path, model, device):
    # 使用 V2 版预处理 (Normalize + Crop) 保证信号强度
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        print(f"\nTesting: {os.path.basename(img_path)}")
        with torch.no_grad():
            output = model(img_t)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            conf = prob[0][pred].item()
            
            # 这里通常 0=Real, 1=Fake，但也取决于数据集定义
            # 我们主要看 Confidence 有没有变化
            label_str = "FAKE" if pred == 1 else "REAL" 
            print(f"   ✅ Prediction: Class {pred} ({label_str}) | Conf: {conf:.4f}")
    except Exception as e:
        print(f"Error: {e}")

def find_first_image(directory):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    for ext in exts:
        files = glob.glob(os.path.join(directory, "**", ext), recursive=True)
        if files:
            return files[0]
    return None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using device: {device}")

    model = SAFE_MB_Infer().to(device)
    model.eval()

    ckpt_path = "results/SAFE_reproduction/20260113_130951_SAFE_MultiBand_SAFE_MB_Attention/checkpoint-best.pth"
    print(f"Loading checkpoint...")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    print("✅ Model loaded successfully!")
    print("\n⚠️  WARNING: Manually boosting High-Frequency (HH) Attention by +5.0! \n")
    
    # 1. Test Deepfake (重点关注这个！)
    df_path = find_first_image("datasets/test1_ForenSynths/test")
    if df_path: run_test(df_path, model, device)
    else: print("❌ No Deepfake image found.")

    # 2. Test Diffusion
    diff_path = find_first_image("datasets/test3_Ojha/test")
    if diff_path: run_test(diff_path, model, device)
    else: print("❌ No Diffusion image found.")

    # 3. Test ProGAN/Real
    real_path = find_first_image("datasets/train_ForenSynths/val")
    if real_path: run_test(real_path, model, device)
    else: print("❌ No Real/ProGAN image found.")
