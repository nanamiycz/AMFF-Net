import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torchvision import transforms
from timm.models.registry import register_model

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# 带有探针的注意力模块
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
        
        # [PROBE] 抓取数据
        if not self.training:
            avg_weights = y.mean(dim=0).view(-1)
            if avg_weights.shape[0] == 9:
                w_lh = avg_weights[0:3].mean().item()
                w_hl = avg_weights[3:6].mean().item()
                w_hh = avg_weights[6:9].mean().item()
                print(f"\n�� [Analysis Probe] LH: {w_lh:.4f} | HL: {w_hl:.4f} | HH: {w_hh:.4f}")
        
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

class SAFE_MB(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False, ablation_mode='ours', **kwargs):
        super(SAFE_MB, self).__init__()
        self.ablation_mode = ablation_mode
        self.inplanes = 64
        
        if self.ablation_mode == 'baseline':
            freq_channels = 3
            self.use_attention = False
        else:
            freq_channels = 9
            self.use_attention = (self.ablation_mode == 'ours')

        if self.use_attention:
            self.freq_attention = AdaptiveFrequencyAttention(channel=freq_channels, reduction=3)
        
        # [关键修复 1] 这里输出必须是 3 (匹配 Checkpoint [3, 9, 1, 1])
        # 之前的代码写的是 64，导致了 Size Mismatch
        self.freq_fusion = nn.Conv2d(freq_channels, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.freq_fusion.weight, mode='fan_out', nonlinearity='relu')

        # [关键修复 2] 这里的 conv1 是 3x3，且必须存在，因为融合后变成了 3 通道，需要它变成 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def _preprocess_dwt(self, x):
        from pytorch_wavelets import DWTForward
        if not hasattr(self, 'dwt_fn'):
            self.dwt_fn = DWTForward(J=1, mode='symmetric', wave='bior1.3').to(x.device)
        try:
            Yl, Yh = self.dwt_fn(x)
        except:
            self.dwt_fn = DWTForward(J=1, mode='symmetric', wave='bior1.3').to(x.device)
            Yl, Yh = self.dwt_fn(x)
        lh, hl, hh = Yh[0][:, :, 0, :, :], Yh[0][:, :, 1, :, :], Yh[0][:, :, 2, :, :]
        x_freq = torch.cat([lh, hl, hh], dim=1)
        return F.interpolate(x_freq, size=(x.shape[-2]//2, x.shape[-1]//2), mode='bilinear', align_corners=False)

    def forward(self, x):
        x_freq = self._preprocess_dwt(x)
        if self.use_attention:
            x_freq = self.freq_attention(x_freq)
        
        # [关键修复 3] 恢复正确的前向传播流程
        x = self.freq_fusion(x_freq) # 9 -> 3 通道
        x = self.conv1(x)            # 3 -> 64 通道 (使用 3x3 卷积)
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

def ResNet(*args, **kwargs): pass
def resnet34(*args, **kwargs): pass
def resnet50(*args, **kwargs): pass
def resnet101(*args, **kwargs): pass
def resnet152(*args, **kwargs): pass

@register_model
def resnet18(pretrained=False, **kwargs):
    return SAFE_MB(BasicBlock, [2, 2, 2, 2], ablation_mode='ours', **kwargs)
