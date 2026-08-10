import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class VITGSMAP(nn.Module):
    def __init__(self, config, out_ch=192, keep_blocks=2):
        super().__init__()
        self.config = config
        self.out_ch = out_ch

        vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            in_chans=self.config.MODEL.TIME_STEP,
            num_classes=0,
            global_pool="",          # vẫn giữ, nhưng ta sẽ KHÔNG dùng vit(x) nữa
            features_only=False,
            dynamic_img_size=True,
        )

        # cắt số block bạn muốn
        vit.blocks = vit.blocks[:keep_blocks]
        self.vit = vit

        # vit_tiny embed_dim thường = 192
        embed_dim = getattr(self.vit, "embed_dim", 192)

        # project về out_ch nếu cần
        self.proj = nn.Conv2d(embed_dim, out_ch, kernel_size=1, bias=False)

    def _tokens_to_grid(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        tokens: (B, N, C) hoặc (B, 1+N, C) (có CLS)
        => (B, C, gh, gw)
        """
        B, N, C = tokens.shape

        # bỏ CLS nếu có
        # gh*gw = (H/patch)*(W/patch)
        ps = self.vit.patch_embed.patch_size
        if isinstance(ps, tuple):
            ph, pw = ps
        else:
            ph = pw = ps

        gh, gw = H // ph, W // pw
        expected = gh * gw

        if N == expected + 1:
            tokens = tokens[:, 1:, :]  # drop CLS
            N = expected
        elif N != expected:
            # fallback: thử drop CLS nếu N lệch 1
            if N - 1 == expected:
                tokens = tokens[:, 1:, :]
                N = expected
            else:
                raise ValueError(f"Token count không khớp: N={N}, expected={expected} (gh={gh}, gw={gw}).")

        x = tokens.transpose(1, 2).reshape(B, C, gh, gw).contiguous()
        return x

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return tokens: (B, N, C) (thường có CLS trong N)
        """
        x = self.vit.patch_embed(x)  # (B, N, C)

        # timm mới có _pos_embed (handle interpolate pos embed + add CLS)
        if hasattr(self.vit, "_pos_embed"):
            x = self.vit._pos_embed(x)
        else:
            # fallback cho timm cũ
            if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
                cls = self.vit.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
            if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
                x = x + self.vit.pos_embed
            if hasattr(self.vit, "pos_drop"):
                x = self.vit.pos_drop(x)

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)
        return x

    def forward(self, x):
        """
        x: (B, T, H, W)
        return: (B, out_ch, H, W)  (mình trả NCHW cho dễ dùng conv)
        """
        B, T, H, W = x.shape

        # resize về 32x32 (chia hết cho patch16)
        x32 = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        tokens = self._forward_tokens(x32)          # (B, N, C) hoặc (B, 1+N, C)
        feat32 = self._tokens_to_grid(tokens, 32, 32)  # (B, C, 2, 2) với patch16@32

        feat32 = self.proj(feat32)                 # (B, out_ch, 2, 2)
        feat = F.interpolate(feat32, size=(self.config.DATA.HEIGHT, self.config.DATA.WIDTH), mode="bilinear", align_corners=False)

        return feat.permute(0, 2, 3, 1)  # (B, out_ch, H, W)
