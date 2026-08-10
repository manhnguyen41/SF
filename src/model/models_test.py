import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .layers import Combined_Spatial, TemporalExactor, PredictionHead, SpatialExactor2, TemporalExactorSTrans
from .stransformer import PatchEmbedding, PositionEmbedding, WindowMultiHeadAttention, UpsampleWithTransposedConv, SEResNet

class PromptMixin:
    def __init__(self, config, hidden_dim):
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        self.max_delta_t = config.MODEL.TEMPORAL.MAX_DELTA_T
        self.hidden_dim = hidden_dim

        if self.prompt_type == 0:
            self.delta_t = nn.Parameter(torch.randn(self.max_delta_t, self.hidden_dim))
        else:
            raise ValueError(f"Wrong prompt_type: {self.prompt_type}")

    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type not in [0, 1]:
                raise ValueError(f"Wrong adding type value: {self.add_type}")
            for lt in lead_time:
                lt = int(lt) - 7
                if lt < 0 or lt >= self.max_delta_t:
                    raise IndexError(f"lead_time {lt + 7} (adjusted: {lt}) out of range for delta_t length {self.max_delta_t}")
                corress_prompt = self.delta_t[lt]  # (hidden_dim,)
                B, H, W, D = temporal_embedding.shape
                corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # (1,1,hidden_dim)
                corress_prompt = corress_prompt.expand(H, W, -1)  # (H,W,hidden_dim)
                list_prompt.append(corress_prompt)

            add_prompt = torch.stack(list_prompt, 0)  # (batch_size, H, W, hidden_dim)

            if self.add_type == 0:
                return temporal_embedding + add_prompt
            else:  # self.add_type == 1
                return torch.cat([temporal_embedding, add_prompt], dim=-1)
        else:
            raise ValueError(f"Wrong prompt type value: {self.prompt_type}")

class PatchMixin:
    def __init__(self, config, embed_dim):
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = embed_dim
        self.config = config

    def cal_num_patches(self, img_size):
        h, w = img_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        return (padded_h // self.patch_size) * (padded_w // self.patch_size)

class Model_Ver1(nn.Module, PromptMixin):
    def __init__(self, config):
        nn.Module.__init__(self)
        PromptMixin.__init__(self, config, config.MODEL.TEMPORAL.HIDDEN_DIM)
        self.config = config

        if config.MODEL.SPATIAL.TYPE == 0:
            self.spatial_exactor = Combined_Spatial(
                in_channels=config.MODEL.IN_CHANNEL,
                out_channels=config.MODEL.SPATIAL.OUT_CHANNEL,
                kernel_sizes=config.MODEL.SPATIAL.KERNEL_SIZES,
                use_batch_norm=config.MODEL.SPATIAL.USE_BATCH_NORM
            )
            temporal_input_size = len(config.MODEL.SPATIAL.KERNEL_SIZES) * config.MODEL.SPATIAL.OUT_CHANNEL
            self.temporal_exactor = TemporalExactor(
                input_size=temporal_input_size,
                hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                num_layers=config.MODEL.TEMPORAL.NUM_LAYERS
            )
        elif config.MODEL.SPATIAL.TYPE == 1:
            self.spatial_exactor = SpatialExactor2(
                in_channels=config.MODEL.IN_CHANNEL,
                out_channels=config.MODEL.SPATIAL.OUT_CHANNEL,
                kernel_size=3,
                use_batch_norm=config.MODEL.SPATIAL.USE_BATCH_NORM,
                num_conv_layers=config.MODEL.SPATIAL.NUM_LAYERS
            )
            self.temporal_exactor = TemporalExactor(
                input_size=config.MODEL.SPATIAL.OUT_CHANNEL,
                hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                num_layers=config.MODEL.TEMPORAL.NUM_LAYERS
            )
        else:
            raise ValueError(f"Wrong spatial type: {config.MODEL.SPATIAL.TYPE}")

        self.prediction_head = PredictionHead(
            config.MODEL.TEMPORAL.HIDDEN_DIM,
            use_layer_norm=config.MODEL.USE_LAYER_NORM,
            dropout=config.MODEL.DROPOUT
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        ncmwf, lead_time = x
        spatial_embedding = self.spatial_exactor(ncmwf)
        temporal_embedding = self.temporal_exactor(spatial_embedding)
        output = self.add_prompt_vecs(temporal_embedding, lead_time)
        output = self.prediction_head(output)
        return output


class SwinTransformerBase(nn.Module, PromptMixin, PatchMixin):
    def __init__(self, config, embed_dim):
        nn.Module.__init__(self)
        PromptMixin.__init__(self, config, config.MODEL.TEMPORAL.HIDDEN_DIM)
        PatchMixin.__init__(self, config, embed_dim)
        self.config = config
        self.dropout = config.MODEL.DROPOUT

        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, embed_dim)
        self.window_attention = WindowMultiHeadAttention(
            embed_dim,
            config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE,
            config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
            config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS,
            config.MODEL.SWIN_TRANSFORMER.FF_DIM,
            self.dropout,
        )
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding(num_patches, embed_dim)
        self.temporal_exactor = TemporalExactorSTrans(embed_dim, config.MODEL.TEMPORAL.HIDDEN_DIM, config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS)
        self.upsample = UpsampleWithTransposedConv(config.MODEL.TEMPORAL.HIDDEN_DIM, embed_dim, scale_factor=self.patch_size)
        self.prediction_head = PredictionHead(embed_dim, use_layer_norm=config.MODEL.USE_LAYER_NORM, dropout=self.dropout)

    def forward_patch_processing(self, x, lead_time):
        batch_size, n_ts, n_ft, h, w = x.shape

        # Combine time and feature dims for patch embedding
        x = x.view(batch_size * n_ts, n_ft, h, w)

        # Pad input to multiple of patch size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = h + pad_h, w + pad_w

        x = self.patch_embed(x)
        x = self.pos_embed(x)

        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size

        x = x.view(batch_size * n_ts, h_patch, w_patch, self.embed_dim)
        x = self.window_attention(x)
        x = x.view(batch_size, n_ts, h_patch, w_patch, -1)
        x = self.temporal_exactor(x)
        x = self.add_prompt_vecs(x, lead_time)
        x = self.upsample(x)
        x = x[:, :h, :w, :]
        x = self.prediction_head(x)
        return x


class SwinTransformer(SwinTransformerBase):
    def __init__(self, config):
        super().__init__(config, embed_dim=config.MODEL.SWIN_TRANSFORMER.EMBED_DIM)


class Model_Ver2(Model_Ver1):
    def __init__(self, config):
        super().__init__(config)
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        self.channel_attn2 = SEResNet(in_channels=192, out_channels=192, reduction_ratio=16)
        self.channel_attn3 = SEResNet(in_channels=128, out_channels=128, reduction_ratio=16)

    def forward(self, x):
        ncmwf, lead_time = x
        batch_size, n_ts, n_ft, h, w = ncmwf.shape
        ncmwf = ncmwf.view(batch_size * n_ts, n_ft, h, w)
        ncmwf = self.channel_attn(ncmwf)
        ncmwf = ncmwf.view(batch_size, n_ts, n_ft, h, w)

        spatial_embedding = self.spatial_exactor(ncmwf)
        temporal_embedding = self.temporal_exactor(spatial_embedding)
        output = self.add_prompt_vecs(temporal_embedding, lead_time)
        output = self.prediction_head(output)
        return output


class SwinTransformer_Ver2(SwinTransformerBase):
    def __init__(self, config):
        super().__init__(config, embed_dim=config.MODEL.SWIN_TRANSFORMER.EMBED_DIM)
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)

    def forward(self, x):
        lead_time = x[1]
        x = x[0]

        batch_size, n_ts, n_ft, h, w = x.shape
        x = x.view(batch_size * n_ts, n_ft, h, w)
        x = self.channel_attn(x)

        # Pad input to multiple of patch size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = h + pad_h, w + pad_w

        x = self.patch_embed(x)
        x = self.pos_embed(x)

        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size

        x = x.view(batch_size * n_ts, h_patch, w_patch, self.embed_dim)
        x = self.window_attention(x)

        x = x.view(batch_size, n_ts, h_patch, w_patch, -1)
        x = self.temporal_exactor(x)
        x = self.add_prompt_vecs(x, lead_time)

        x = self.upsample(x)
        x = x[:, :h, :w, :]
        x = self.prediction_head(x)
        return x


class SwinTransformer_Ver3(SwinTransformerBase):
    def __init__(self, config):
        embed_dim = 768
        super().__init__(config, embed_dim)
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.spatial_encoder = vit.blocks
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        lead_time = x[1]
        x = x[0]

        batch_size, n_ts, n_ft, h, w = x.shape
        x = x.view(batch_size * n_ts, n_ft, h, w)

        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = h + pad_h, w + pad_w

        x = self.patch_embed(x)
        x = self.pos_embed(x)

        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size

        x = self.spatial_encoder(x)

        x = x.view(batch_size, n_ts, h_patch, w_patch, -1)
        x = self.temporal_exactor(x)
        x = self.add_prompt_vecs(x, lead_time)

        x = self.upsample(x)
        x = x[:, :h, :w, :]
        x = self.prediction_head(x)
        return x


class SwinTransformer_Ver4(SwinTransformer_Ver3):
    def __init__(self, config):
        super().__init__(config)
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)

    def forward(self, x):
        lead_time = x[1]
        x = x[0]

        batch_size, n_ts, n_ft, h, w = x.shape
        x = x.view(batch_size * n_ts, n_ft, h, w)
        x = self.channel_attn(x)

        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = h + pad_h, w + pad_w

        x = self.patch_embed(x)
        x = self.pos_embed(x)

        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size

        x = self.spatial_encoder(x)

        x = x.view(batch_size, n_ts, h_patch, w_patch, -1)
        x = self.temporal_exactor(x)
        x = self.add_prompt_vecs(x, lead_time)

        x = self.upsample(x)
        x = x[:, :h, :w, :]
        x = self.prediction_head(x)
        return x