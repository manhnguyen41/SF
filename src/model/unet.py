import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- tiện ích ----------
def kaiming_conv(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

# --------- Depthwise-Separable block ----------
class DWSeparable(nn.Module):
    """
    Depthwise 3x3 (groups=in_ch) + Pointwise 1x1, với GN + GELU.
    """
    def __init__(self, in_ch, out_ch, norm='gn'):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.n1 = nn.GroupNorm(8, in_ch) if norm=='gn' else nn.BatchNorm2d(in_ch)
        self.act1 = nn.GELU()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.n2 = nn.GroupNorm(8, out_ch) if norm=='gn' else nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU()
        self.apply(kaiming_conv)

    def forward(self, x):
        x = self.dw(x); x = self.act1(self.n1(x))
        x = self.pw(x); x = self.act2(self.n2(x))
        return x

class MobileResBlock(nn.Module):
    """
    Residual pre-activation bằng các khối depthwise-separable.
    Có Dropout2d + DropPath để regularize.
    """
    def __init__(self, in_ch, out_ch, norm='gn', p_drop=0.15, p_droppath=0.05):
        super().__init__()
        self.c1 = DWSeparable(in_ch, out_ch, norm=norm)
        self.c2 = DWSeparable(out_ch, out_ch, norm=norm)
        self.skip = nn.Identity() if in_ch==out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        kaiming_conv(self.skip)
        # zero-init lớp cuối để gần identity lúc đầu
        nn.init.zeros_(self.c2.pw.weight)
        self.drop = nn.Dropout2d(p=p_drop)
        self.dp   = DropPath(p=p_droppath)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.drop(out)
        out = self.dp(out)
        return out + self.skip(x)

class MobileUpBlock(nn.Module):
    """
    Upsample (bilinear) + proj 1x1 + concat skip + MobileResBlock
    """
    def __init__(self, in_ch, skip_ch, out_ch, norm='gn', p_drop=0.15, p_droppath=0.05):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False); kaiming_conv(self.proj)
        self.conv = MobileResBlock(out_ch + skip_ch, out_ch, norm=norm,
                                   p_drop=p_drop, p_droppath=p_droppath)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder32(nn.Module):
    """
    Phiên bản Mobile (depthwise-separable) của UNetDecoder32:
      - Input: feats = [f0,f1,f2,f3] (8×8, 4×4, 2×2, 1×1)
      - Output: (B, out_ch, 32, 32)
    """
    def __init__(self, in_channels_list, dec_channels=64, out_ch=1,
                 norm='gn', out_activation='softplus'):
        super().__init__()
        C0,C1,C2,C3 = in_channels_list
        dec = dec_channels

        # bottleneck 1x1 → dec
        self.bot = MobileResBlock(C3, dec, norm=norm, p_drop=0.10, p_droppath=0.05)
        # up path tới 8×8
        self.up2 = MobileUpBlock(dec, C2, dec, norm=norm, p_drop=0.10, p_droppath=0.05)  # 1->2
        self.up1 = MobileUpBlock(dec, C1, dec, norm=norm, p_drop=0.10, p_droppath=0.05)  # 2->4
        self.up0 = MobileUpBlock(dec, C0, dec, norm=norm, p_drop=0.10, p_droppath=0.05)  # 4->8

        # 8→16→32 (mobile residual)
        self.upx1 = MobileResBlock(dec, dec, norm=norm, p_drop=0.15, p_droppath=0.05)   # 8->16
        self.upx2 = MobileResBlock(dec, dec, norm=norm, p_drop=0.20, p_droppath=0.05)   # 16->32

        # Dropout ở high-res thêm (tuỳ chọn)
        self.drop16 = nn.Dropout2d(p=0.15)
        self.drop32 = nn.Dropout2d(p=0.25)

        self.head = nn.Conv2d(dec, out_ch, 1, bias=True); kaiming_conv(self.head)
        self.out_act = nn.Softplus() if out_activation=='softplus' else nn.Identity()

    def forward(self, feats, baseline=None):
        f0,f1,f2,f3 = feats
        x = self.bot(f3)         # 1x1
        x = self.up2(x, f2)      # -> 2x2
        x = self.up1(x, f1)      # -> 4x4
        x = self.up0(x, f0)      # -> 8x8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 8->16
        x = self.upx1(x); x = self.drop16(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 16->32
        x = self.upx2(x); x = self.drop32(x)

        y_delta = self.out_act(self.head(x))   # dự đoán Δ (nếu dùng Softplus ⇒ Δ≥0)

        # Global residual (tuỳ chọn)
        if baseline is not None:
            if baseline.shape[-2:] != (32,32):
                baseline = F.interpolate(baseline, size=(32,32), mode='nearest')
            if baseline.size(1) == 1 and y_delta.size(1) > 1:
                baseline = baseline.expand(-1, y_delta.size(1), -1, -1)
            return y_delta + baseline
        return y_delta
