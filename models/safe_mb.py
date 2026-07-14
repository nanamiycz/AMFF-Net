import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torchvision import transforms
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# [New] Frequency Attention Module (AFA)
class AdaptiveFrequencyAttention(nn.Module):
    def __init__(self, channel=9, reduction=3):
        super(AdaptiveFrequencyAttention, self).__init__()
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
        # Dynamically adjust the weight of each frequency band
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ==========================================
# Main Model: SAFE_MB supporting Ablation Study
# ==========================================
class SAFE_MB(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False, ablation_mode='ours'):
        super(SAFE_MB, self).__init__()
        
        self.ablation_mode = ablation_mode
        print(f"🏗️ Initializing SAFE_MB Model Mode: {self.ablation_mode.upper()}")

        self.inplanes = 64
        
        # 1. Frequency Processing Logic
        # ---------------------------------------------------
        if self.ablation_mode == 'baseline':
            freq_channels = 3  # Only HH
            self.use_attention = False
        elif self.ablation_mode == 'll_only':
            freq_channels = 3  # Only LL
            self.use_attention = False
        elif self.ablation_mode == 'lh_hl_only':
            freq_channels = 6  # Only LH + HL
            self.use_attention = False
        elif self.ablation_mode == 'all_bands':
            freq_channels = 12 # LL + LH + HL + HH
            self.use_attention = False # Concat directly to show interference
        elif self.ablation_mode == 'naive':
            freq_channels = 9  # LH + HL + HH without attention
            self.use_attention = False
        elif self.ablation_mode == 'ours':
            freq_channels = 9  # LH + HL + HH with attention
            self.use_attention = True
        else:
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}")

        # 2. Attention Module (Only initialized for Ours)
        if self.use_attention:
            self.freq_attention = AdaptiveFrequencyAttention(channel=freq_channels, reduction=3)
        
        # 3. Fusion Layer (1x1 Conv)
        self.freq_fusion = nn.Conv2d(freq_channels, 64, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.freq_fusion.weight, mode='fan_out', nonlinearity='relu')
        # ---------------------------------------------------

        # Standard ResNet Backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False) # Skipped in forward
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        from pytorch_wavelets import DWTForward
        if not hasattr(self, 'dwt_fn'):
            self.dwt_fn = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
            
        Yl, Yh = self.dwt_fn(x)
        
        ll = Yl
        lh = Yh[0][:, :, 0, :, :]
        hl = Yh[0][:, :, 1, :, :]
        hh = Yh[0][:, :, 2, :, :]
        
        if self.ablation_mode == 'baseline':
            x_freq = hh
        elif self.ablation_mode == 'll_only':
            x_freq = ll
        elif self.ablation_mode == 'lh_hl_only':
            x_freq = torch.cat([lh, hl], dim=1)
        elif self.ablation_mode == 'all_bands':
            x_freq = torch.cat([ll, lh, hl, hh], dim=1)
        elif self.ablation_mode in ['naive', 'ours']:
            x_freq = torch.cat([lh, hl, hh], dim=1)
        
        return transforms.Resize([x.shape[-2]//2, x.shape[-1]//2])(x_freq) 
        
    def forward(self, x):
        # 1. DWT & Band Selection
        x_freq = self._preprocess_dwt(x)
        
        # 2. Attention (Only for 'ours')
        if self.use_attention:
            x_freq = self.freq_attention(x_freq)
        
        # 3. Fusion (N -> 64)
        x = self.freq_fusion(x_freq)
        
        # 4. Backbone flow
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

def resnet18(pretrained=False, ablation_mode='ll_only', **kwargs):
    """Constructs a ResNet-18 model with ablation support."""
    # ↓↓↓ 就是这一行，千万不能丢！↓↓↓
    model = SAFE_MB(BasicBlock, [2, 2, 2, 2], ablation_mode=ablation_mode, **kwargs)
    
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model