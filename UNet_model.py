import torch
import torch.nn as nn

# =====================
# 3. 网络结构
# =====================
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(GeneratorUNet, self).__init__()

        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.encoder = nn.ModuleList([
            down_block(in_channels, 64),
            down_block(64, 128),
            down_block(128, 256),
            down_block(256, 512),
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU()
        )

        self.decoder = nn.ModuleList([
            up_block(512, 512),
            up_block(1024, 256),
            up_block(512, 128),
            up_block(256, 64),
        ])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        for i, up in enumerate(self.decoder):
            x = up(x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)

        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, 1, 0)),
        )

    def forward(self, img):
        return self.model(img)
    


