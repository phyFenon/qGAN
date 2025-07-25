import torch
import torch.nn as nn
import torch.nn.functional as F

# 门控卷积：输出为 feature * gate（gate使用Sigmoid激活）
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(GatedConv2d, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.feature_conv(x)
        gate = self.sigmoid(self.gating_conv(x))
        return feature * gate

# 通道注意力模块（Squeeze-and-Excitation）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)     # 压缩为 [B, C]
        w = self.fc(w).view(b, c, 1, 1) # 扩展为 [B, C, 1, 1]
        return x * w                    # 通道注意力加权

# 改进后的 Gated-UNet
class GatedUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(GatedUNet, self).__init__()

        # 编码器模块（使用门控卷积 + 通道注意力）
        def down_block(in_c, out_c, dilation=1):
            return nn.Sequential(
                GatedConv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                SEBlock(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # 解码器模块（反卷积 + BN + ReLU + 可选Dropout）
        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.3))
            return nn.Sequential(*layers)

        # 编码器4层
        self.enc1 = down_block(in_channels, 64)
        self.enc2 = down_block(64, 128)
        self.enc3 = down_block(128, 256)
        self.enc4 = down_block(256, 512)

        # 瓶颈层（使用膨胀卷积扩大感受野）
        self.bottleneck = nn.Sequential(
            GatedConv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        # 解码器4层（使用 skip connection）
        self.dec1 = up_block(512, 512, dropout=True)
        self.dec2 = up_block(512 + 512, 256)
        self.dec3 = up_block(256 + 256, 128)
        self.dec4 = up_block(128 + 128, 64)

        # 输出层：输出通道数为3，使用Tanh归一化到 [-1, 1]
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    # 前向传播过程
    def forward(self, x):
        e1 = self.enc1(x)                     # 编码器第1层
        e2 = self.enc2(e1)                    # 编码器第2层
        e3 = self.enc3(e2)                    # 编码器第3层
        e4 = self.enc4(e3)                    # 编码器第4层

        bottleneck = self.bottleneck(e4)      # 瓶颈层


        d1 = self.dec1(bottleneck)
        e4_up = F.interpolate(e4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d1, e4_up], dim=1))

        e3_up = F.interpolate(e3, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d2, e3_up], dim=1))

        e2_up = F.interpolate(e2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d3, e2_up], dim=1))

        e1_up = F.interpolate(e1, size=d4.shape[2:], mode='bilinear', align_corners=False)
        output = self.final(torch.cat([d4, e1_up], dim=1))
        
        return output



class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            # 最后一层：输出为 [B, 1, h, w]
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1))

        )

    def forward(self, x):
        return self.model(x)  # 输出维度为 [B, 1, h, w]

