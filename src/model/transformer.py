import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== Blocks cơ bản ===============
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        )
        nn.init.kaiming_normal_(self.block[-1].weight, nonlinearity='relu')
    def forward(self, x): return self.block(x)

class ResPreAct(nn.Module):
    """Residual pre-activation: GN→GELU→Conv ×2 + skip"""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.c1 = ConvGNAct(in_ch, out_ch, 3, 1, 1, groups)
        self.c2 = ConvGNAct(out_ch, out_ch, 3, 1, 1, groups)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity='relu')
        # gần identity
        nn.init.zeros_(self.c2.block[-1].weight)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        return out + self.skip(x)

class UpBlock(nn.Module):
    """Upsample (bilinear) + proj 1×1 + concat skip + ResPreAct"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
        self.conv = ResPreAct(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# =============== Cross-Attention 2D ===============
class CrossAttn2D(nn.Module):
    """
    Q: (B, Cq, Hq, Wq)  — query map (decoder hiện tại)
    K/V: list các feature encoder đa tỉ lệ; mỗi cái sẽ được proj→D, flatten và concat.
    Dùng nn.MultiheadAttention (batch_first=True).
    """
    def __init__(self, c_q, c_k_list, dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.q_proj = nn.Conv2d(c_q, dim, 1, bias=False)
        self.kv_proj = nn.ModuleList([nn.Conv2d(c, dim, 1, bias=False) for c in c_k_list])
        for m in [self.q_proj, *self.kv_proj]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Conv2d(dim, c_q, 1, bias=False),
            nn.GroupNorm(8, c_q)
        )
        nn.init.kaiming_normal_(self.out[0].weight, nonlinearity='relu')

        # learnable positional encodings (nhẹ nhàng)
        self.pos_q = None
        self.pos_k = None
        self.dim = dim

    def _pos(self, B, L, device):
        # sinusoid đơn giản: (B,L,dim)
        if (self.pos_q is None) or (self.pos_q.shape[1] < L):
            pos = torch.arange(L, device=device).float()[None, :, None]
            i = torch.arange(self.dim, device=device).float()[None, None, :]
            div = torch.exp(- (2 * (i//2)) * torch.log(torch.tensor(10000.0, device=device)) / self.dim)
            pe = torch.zeros(1, L, self.dim, device=device)
            pe[..., 0::2] = torch.sin(pos * div[..., 0::2])
            pe[..., 1::2] = torch.cos(pos * div[..., 1::2])
            return pe
        return self.pos_q[:, :L, :]

    def forward(self, q_map, enc_feats):
        """
        q_map:  (B, Cq, Hq, Wq)
        enc_feats: list T feature encoder; mỗi cái sẽ proj->dim, flatten và concat làm K/V.
        """
        B, Cq, Hq, Wq = q_map.shape
        device = q_map.device

        # Q tokens
        q = self.q_proj(q_map)                               # (B, D, Hq, Wq)
        q = q.flatten(2).transpose(1, 2)                     # (B, Lq=Hq*Wq, D)

        # K/V tokens từ multi-scale encoder
        kv_list = []
        for i, f in enumerate(enc_feats):
            kv = self.kv_proj[i](f)                          # (B, D, H, W)
            kv = kv.flatten(2).transpose(1, 2)               # (B, L_i, D)
            kv_list.append(kv)
        kv = torch.cat(kv_list, dim=1)                       # (B, Lk, D)

        # pos enc (đơn giản)
        pos_q = self._pos(B, q.shape[1], device)
        pos_k = self._pos(B, kv.shape[1], device)

        q_in = q + pos_q
        k_in = kv + pos_k
        v_in = kv

        # MHA expects (B, L, D)
        attn_out, _ = self.mha(q_in, k_in, v_in)             # (B, Lq, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, self.dim, Hq, Wq)
        out = self.out(attn_out) + q_map                     # resid về Cq
        return out

# =============== Decoder: Cross-Attn @8×8 & @32×32 ===============
class CrossAttnDecoder32(nn.Module):
    """
    Input: feats = [f0,f1,f2,f3] (8×8, 4×4, 2×2, 1×1)
    Output: (B, out_ch, 32, 32)
    - Up tới 8×8, cross-attn @8×8.
    - Lên 32×32, cross-attn @32×32 (K/V là encoder upsample về 32×32).
    """
    def __init__(self, in_channels_list, dec_channels=256, out_ch=1,
                 num_heads=8, norm_groups=8, out_activation='softplus'):
        super().__init__()
        C0, C1, C2, C3 = in_channels_list
        dec = dec_channels

        # Bottleneck từ f3
        self.bot = ResPreAct(C3, dec, groups=norm_groups)

        # Up path tới 8×8
        self.up2 = UpBlock(dec, C2, dec)  # 1->2
        self.up1 = UpBlock(dec, C1, dec)  # 2->4
        self.up0 = UpBlock(dec, C0, dec)  # 4->8

        # Cross-Attn @8×8 (Q: dec, K/V: [f0,f1,f2,f3] đã proj)
        self.ca8 = CrossAttn2D(c_q=dec, c_k_list=[C0, C1, C2, C3],
                               dim=dec, num_heads=num_heads)

        # Lên 32×32 (8→16→32)
        self.up16 = ResPreAct(dec, dec, groups=norm_groups)
        self.up32 = ResPreAct(dec, dec, groups=norm_groups)

        # Cross-Attn @32×32: dùng encoder đã upsample -> 32×32 làm K/V
        self.ca32 = CrossAttn2D(c_q=dec, c_k_list=[dec, dec, dec, dec],  # sẽ feed 4 bản 32×32 đã proj
                                dim=dec, num_heads=num_heads)

        # Chuẩn hoá encoder lên 32×32 trước khi đưa vào ca32
        self.enc32_proj = nn.ModuleList([
            nn.Conv2d(C0, dec, 1, bias=False),
            nn.Conv2d(C1, dec, 1, bias=False),
            nn.Conv2d(C2, dec, 1, bias=False),
            nn.Conv2d(C3, dec, 1, bias=False),
        ])
        for m in self.enc32_proj:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        self.head = nn.Conv2d(dec, out_ch, 1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity='relu')
        self.out_act = nn.Softplus() if out_activation=='softplus' else nn.Identity()

    def forward(self, feats):
        f0, f1, f2, f3 = feats  # 8×8, 4×4, 2×2, 1×1

        # bot → 1×1
        x = self.bot(f3)

        # lên 8×8
        x = self.up2(x, f2)   # 2×2
        x = self.up1(x, f1)   # 4×4
        x = self.up0(x, f0)   # 8×8

        # cross-attn @8×8 dùng multi-scale encoder (gốc)
        x = self.ca8(x, [f0, f1, f2, f3])

        # 8→16→32 (residual conv ở mỗi bước)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 16×16
        x = self.up16(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 32×32
        x = self.up32(x)

        # chuẩn bị K/V @32×32: upsample các encoder map về 32×32 rồi proj→dec
        e0 = F.interpolate(f0, size=(32,32), mode='bilinear', align_corners=False)
        e1 = F.interpolate(f1, size=(32,32), mode='bilinear', align_corners=False)
        e2 = F.interpolate(f2, size=(32,32), mode='bilinear', align_corners=False)
        e3 = F.interpolate(f3, size=(32,32), mode='bilinear', align_corners=False)
        e0 = self.enc32_proj[0](e0); e1 = self.enc32_proj[1](e1)
        e2 = self.enc32_proj[2](e2); e3 = self.enc32_proj[3](e3)

        # cross-attn @32×32
        x = self.ca32(x, [e0, e1, e2, e3])

        y = self.head(x)
        return self.out_act(y)
