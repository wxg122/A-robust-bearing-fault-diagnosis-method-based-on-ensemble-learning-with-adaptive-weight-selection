import torch
from torch import nn
from torch.nn import init

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        #torch.Size([8, 256, 8, 1])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)#torch.Size([8, 256])
        y = self.fc(y).view(b, c, 1, 1)#torch.Size([8, 256, 1, 1])
        y = y.expand_as(x)

        return y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)#torch.Size([8, 1, 64, 32])
        max_out, _ = torch.max(x, dim=1, keepdim=True)#torch.Size([8, 1, 64, 32])
        out = torch.cat([avg_out, max_out], dim=1)#torch.Size([8, 2, 64, 32])
        out = self.conv(out)#torch.Size([8, 1, 64, 32])
        out = self.sigmoid(out)#torch.Size([8, 1, 64, 32])
        return out
