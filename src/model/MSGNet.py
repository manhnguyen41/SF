import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple

# ==============================
# Hỗ trợ
# ==============================
def extract_blocks(x, block_size=3):
    """
    x: (B, T, H, W)
    Trả về: (B, T, N, block_size*block_size)
    """
    B, T, H, W = x.shape
    
    pad_H = (block_size - H % block_size) % block_size
    pad_W = (block_size - W % block_size) % block_size
    if pad_H > 0 or pad_W > 0:
        x = F.pad(x, (0, pad_W, 0, pad_H))
    H_p, W_p = x.shape[2], x.shape[3]
    x_blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    B, T, h_blocks, w_blocks, _, _ = x_blocks.shape
    N = h_blocks * w_blocks
    x_blocks = x_blocks.contiguous().view(B, T, N, block_size * block_size)
    return x_blocks, H_p, W_p, h_blocks, w_blocks

def sinusoidal_position_embedding(L, d_model):
    """Trả về tensor (1, d_model, L) cho cùng shape với PE hiện tại"""
    position = torch.arange(L, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) *
        (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(L, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.transpose(0, 1).unsqueeze(0)

# ==============================
# Custom MultiheadAttention
# ==============================
class CustomMultiheadAttention(nn.Module):
    def __init__(self, config,embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if self.num_heads > self.embed_dim:
            self.num_heads = self.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim phải chia hết cho num_heads"
        hidden = max(1, config.MODEL.R//2)
        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden, embed_dim, bias=False)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden, embed_dim, bias=False)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden, embed_dim, bias=False)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden, embed_dim, bias=False)
        )
        self.dropout = nn.Dropout(config.MODEL.DROPOUT)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Áp dụng các phép chiếu tuyến tính
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape các tensor để chuẩn bị cho attention
        B, T, E = query.shape
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Áp dụng scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        
        # Concatenate heads và áp dụng lớp chiếu output
        output = output.permute(0, 2, 1, 3).reshape(B, T, E)
        output = self.out_proj(output)
        
        return output, attn

# ==============================
# FFT chọn top-k chu kỳ
# ==============================
class FFTScale(nn.Module):
    def __init__(self, top_k=3):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        B, T, N, F = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amp = torch.mean(torch.abs(xf), dim=(0, 2, 3))
        
        k = min(self.top_k, amp.shape[0] - 1)
        if k <= 0:
            return [1], torch.tensor([1.0], device=x.device)

        _, topk_idx = torch.topk(amp[1:], k)
        topk_idx = topk_idx + 1
        scales = [max(int(T // idx.item()), 1) for idx in topk_idx]
        return scales, amp[topk_idx]

# ==============================
# Adaptive Graph Layer (không thay đổi)
# ==============================
class AdaptiveGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2, P=[1, 2]):
        super().__init__()
        self.P = P
        self.E1 = nn.Parameter(torch.randn(in_dim, hidden_dim))
        self.E2 = nn.Parameter(torch.randn(in_dim, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(len(P) * in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        B, N, D = x.shape
        A = torch.matmul(self.E1, self.E2.T)
        A = torch.relu(A)
        A = F.softmax(A, dim=-1)
        H_list = []
        A_power = torch.eye(D, device=x.device)
        for j in range(1, max(self.P) + 1):
            A_power = torch.matmul(A_power, A)
            if j in self.P:
                Hj = torch.matmul(x, A_power)
                H_list.append(Hj)
        H_concat = torch.cat(H_list, dim=-1)
        return self.mlp(H_concat)

# ==============================
# Block Attention đã bỏ tích hợp LoRA ở đây
# ==============================
class BlockAttention(nn.Module):
    def __init__(self, config,N, block_size=3, d_model=64, n_heads=4):
        super().__init__()
        self.N = N
        self.block_size = block_size
        self.block_features = block_size * block_size
        self.attn_list = nn.ModuleList([
            CustomMultiheadAttention(config, d_model, n_heads) for _ in range(N)
        ])
        self.norm_list = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(N)])
        self.mlp_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, max(1, d_model // 2)),
                nn.LeakyReLU(),                        # hoặc nn.GELU() nếu muốn
                nn.Linear(max(1, d_model // 2), d_model)
            )
            for _ in range(N)
        ])

    def forward(self, x):
        B, T, N, F = x.shape
        
        assert N == self.N, "Số block x phải = N"
        
        out_blocks = []
        for i in range(N):
            x_block = x[:, :, i, :]
            x_block_attn, _ = self.attn_list[i](x_block, x_block, x_block)
            x_block_attn = x_block_attn + x_block # Thêm residual connection
            x_block_attn = self.norm_list[i](x_block_attn)
            x_block_emb = self.mlp_list[i](x_block_attn)
            out_blocks.append(x_block_emb.view(B, T, 1, -1))
        out = torch.cat(out_blocks, dim=2)
        return out

# ==============================
# Scale Graph Block đã bỏ tích hợp LoRA ở đây
# ==============================
class ScaleGraphBlock(nn.Module):
    def __init__(self, config, d_model, n_heads=4, top_k=3):
        super().__init__()
        if n_heads > d_model:
            n_heads = d_model
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads!"
        self.fft_scale = FFTScale(top_k=top_k)
        self.top_k = top_k
        self.d_model = d_model
        self.adp_glayers = nn.ModuleList([
            AdaptiveGraphLayer(d_model, d_model, hidden_dim=d_model) for _ in range(top_k)
        ])
        self.attn_blocks = nn.ModuleList([
            CustomMultiheadAttention(config,d_model, n_heads) for _ in range(top_k)
        ])
        self.norm_blocks = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(top_k)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N, D = x.shape
        scales, weight = self.fft_scale(x)
        scales = scales[:self.top_k]
        weight = weight[:self.top_k]
        
        if len(scales) == 0:
            scales = [1]
            weight = torch.tensor([1.0], device=x.device)
        
        out_scales = []
        max_T_new = 0
        for i, s in enumerate(scales):
            s = max(int(s), 1)
            T_new = T // s if T >= s else 1
            max_T_new = max(max_T_new, T_new)
            x_scaled = x[:, :T_new * s].reshape(B, T_new, s, N, D).mean(dim=2)
            x_graph = self.adp_glayers[i](x_scaled.reshape(B * T_new, N, D))
            x_graph = x_graph.reshape(B, T_new, N, D)
            q = k = v = x_graph.reshape(B, T_new * N, D)
            attn_out, _ = self.attn_blocks[i](q, k, v)
            out = self.norm_blocks[i](attn_out + q)
            out = out.reshape(B, T_new, N, D)
            out_scales.append(out)

        padded_scales = []
        for out in out_scales:
            T_cur = out.shape[1]
            if T_cur < max_T_new:
                pad_size = max_T_new - T_cur
                pad_tensor = torch.zeros(B, pad_size, N, D, device=out.device)
                out = torch.cat([out, pad_tensor], dim=1)
            padded_scales.append(out)

        padded_scales = torch.stack(padded_scales, dim=0)
        weight = F.softmax(weight, dim=0)
        weight = weight.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, B, max_T_new, N, self.d_model)
        xg = weight * padded_scales
        xg = torch.sum(xg, dim=0)
        if xg.shape[1] != T:
            xg_3d = xg.permute(0, 2, 3, 1).reshape(B * N, D, max_T_new)
            xg_upsampled = F.interpolate(xg_3d, size=T, mode='linear', align_corners=False)
            xg = xg_upsampled.reshape(B, N, D, T).permute(0, 3, 1, 2)
        return xg

# ==============================
# TimeSeriesEmbedding
# ==============================
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, in_channels=9, d_model=64, L=10, P=3, alpha=0.5):
        super().__init__()
        self.d_model = d_model
        self.L = L
        self.P = P
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.conv1d = nn.Conv1d(in_channels, d_model, kernel_size=3, stride=1, padding=1)
        pe = sinusoidal_position_embedding(L, d_model)
        self.register_buffer("PE", pe)
        se_list = [sinusoidal_position_embedding(L, d_model) for _ in range(P)]
        self.register_buffer("SE_sum", torch.stack(se_list).sum(dim=0))

    def forward(self, x):
        B, T, N, C = x.shape
        
        x_hat = (x - x.mean(dim=1, keepdim=True))
        x_hat = x_hat.permute(0, 2, 3, 1).reshape(B * N, C, T)
        x_conv = self.alpha * self.conv1d(x_hat)
        
        x_emb = x_conv + self.PE + self.SE_sum
        
        x_emb = x_emb.reshape(B, N, self.d_model, T).permute(0, 3, 1, 2)
        return x_emb

# ==============================
# Full Model - Áp dụng PEFT ở đây
# ==============================
class ScaleGraphModel2D(nn.Module):
    def __init__(self, config, d_model=64, n_blocks=2, n_heads=4, top_k=3, out_channel=192, P=3, block_size=3, L=30):
        super().__init__()
        self.block_size = block_size
        self.L = L
        self.config = config
        self.numblock = ((self.config.DATA.HEIGHT_ESP - 1) // block_size + 1) * ((self.config.DATA.WIDTH_ESP - 1) // block_size + 1)
        self.block_attn = nn.ModuleList([BlockAttention(config=self.config,N=self.numblock, d_model=d_model) for _ in range(n_blocks)])
        self.embedding = TimeSeriesEmbedding(in_channels=block_size * block_size, d_model=d_model, L=self.L, P=P)
        self.blocks = nn.ModuleList([ScaleGraphBlock(self.config, d_model, n_heads=n_heads, top_k=top_k) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(d_model, config.MODEL.TEMPORAL.MAX_DELTA_T)
        self.out_channel = out_channel
        self.H, self.W = self.config.DATA.HEIGHT, self.config.DATA.WIDTH
        self.patch_size = block_size
        padded_h, padded_w = self.cal_padding([self.config.DATA.HEIGHT_ESP, self.config.DATA.WIDTH_ESP])
        self.last = nn.Sequential(
            nn.Linear(padded_h * padded_w, d_model * 4),
            nn.LeakyReLU(),
            nn.Dropout(config.MODEL.DROPOUT),
            
            nn.Linear(d_model * 4, self.config.DATA.HEIGHT * self.config.DATA.WIDTH),
        ) 
        
        self.top_k = top_k
        print("Model ScaleGraphModel2D loaded without direct LoRA integration in sub-modules.")
    def cal_padding(self, img_size):
        # Hàm này vẫn giữ nguyên
        h, w = img_size[0], img_size[1]
        
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        return padded_h, padded_w
    def forward(self, x, leadtime):
        B, T, H, W = x.shape
        
        x_blocks, H_p, W_p, h_blocks, w_blocks = extract_blocks(x, self.block_size)
        
        h = self.embedding(x_blocks)
        
        for i, (bl, blk) in enumerate(zip(self.block_attn, self.blocks)):
            h = bl(h)
            h = blk(h) + h
        out = self.fc_out(h)
        out_blocks = out.reshape(B, T, h_blocks, w_blocks, out.shape[-1])
        out_up = out_blocks.repeat_interleave(self.block_size, dim=2).repeat_interleave(self.block_size, dim=3)
        # out_up = out_up[:, 0, (H_p - self.H) // 2: (H_p - self.H) // 2 + self.H, (W_p - self.W) // 2: (W_p - self.W) // 2 + self.W, :]
        out_up = out_up[:, 0, :, :, :]
        # out_up = self.last(out_up.reshape(B, -1, H_p * W_p)).reshape(B, -1, self.H, self.W)
        out_up = out_up.permute(0, 3, 1, 2)
        
        out_up = self.last(out_up.reshape(B, -1, H_p * W_p)).reshape(B, -1, self.H, self.W)
        # out_up = F.interpolate(
        #     out_up,
        #     size=(self.H, self.W),                   # upsample lên đúng H, W mong muốn
        #     mode="bilinear",                         # hoặc "nearest" nếu là mask / label
        #     align_corners=False
        # )                                            # [B, C, H, W]
        out_up = out_up.permute(0, 2, 3, 1)
        
        out = []
        for i, lt in enumerate(leadtime):
            lt = int(lt)
            lt = (lt-1)
            
            out.append(out_up[i, :, :, lt:lt+1])
        out_up = torch.stack(out, dim = 0)
        return out_up.repeat(1, 1, 1, self.out_channel)

# Bạn sẽ khởi tạo mô hình và áp dụng PEFT ở đây, ví dụ trong file main.py:
#
# from src.model.MSGNet import ScaleGraphModel2D, CustomMultiheadAttention
#
# # Khởi tạo mô hình
# model = ScaleGraphModel2D(config=your_config)
#
# # Cấu hình LoRA
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     # Target các lớp Linear trong mô hình
#     # Ví dụ: "q_proj", "k_proj", "v_proj", "out_proj" trong CustomMultiheadAttention
#     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], 
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.FEATURE_EXTRACTION
# )
#
# # Áp dụng PEFT lên toàn bộ mô hình
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()