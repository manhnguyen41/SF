import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            in_ch // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class UNet(nn.Module):
    """
    Input:
        x[0]: [B, T, C, H, W]

    Output:
        [B, H, W, 1]
    """

    def __init__(self, config):
        super().__init__()

        in_channel = config.MODEL.IN_CHANNEL
        if in_channel is None:
            in_channel = 13

        T = config.MODEL.ECMWF_TIME_STEP
        in_channel *= T

        base = getattr(config.MODEL, "BASE_CHANNEL", 64)

        # Encoder
        self.inc = DoubleConv(in_channel, base)

        self.down1 = Down(base, base * 2)

        self.down2 = Down(base * 2, base * 4)

        self.down3 = Down(base * 4, base * 8)

        # Decoder
        self.up1 = Up(base * 8, base * 4)

        self.up2 = Up(base * 4, base * 2)

        self.up3 = Up(base * 2, base)

        # Head
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):

        x = x[0]                      # [B,T,C,H,W]

        B, T, C, H, W = x.shape

        x = x.view(B, T * C, H, W)

        # Encoder
        x1 = self.inc(x)              # base

        x2 = self.down1(x1)           # base*2

        x3 = self.down2(x2)           # base*4

        x4 = self.down3(x3)           # base*8

        # Decoder
        x = self.up1(x4, x3)

        x = self.up2(x, x2)

        x = self.up3(x, x1)

        y = self.head(x)

        return y.permute(0, 2, 3, 1).contiguous()