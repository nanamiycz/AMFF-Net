import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import glob

# ==========================================
# 1. 模型定义 (保持完全一致)
# ==========================================
class AdaptiveFrequencyAttention(nn.Module):
    def __init__(self, channel=9, reduction=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        if not self.training:
            w = y.mean(dim=0).view(-1).detach().cpu().numpy()
            # 打印醒目的 Probe 数据
            print(f"   📊 [Probe] LH: {w[0:3].mean():.4f} | HL: {w[3:6].mean():.4f} | HH: {w[6:9].mean():.4f}")
            
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
    # ====================================================
    # 关键修改：加入 ImageNet 标准归一化和裁剪
    # 这会极大增强输入信号的强度，激活 Attention！
    # ====================================================
    transform = transforms.Compose([
        transforms.Resize(256),       # 先缩放到 256
        transforms.CenterCrop(224),   # 再切出中间 224 (标准 ResNet 输入)
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
            
        print(f"   ✅ Prediction: Class {pred} (Conf: {conf:.4f})")
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
    
    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 加载状态字典
    model.load_state_dict(checkpoint['model'], strict=False)
    print("✅ Model loaded successfully!")
    
    # 1. Test Deepfake
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
