from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Combined_Spatial, TemporalExactor, PredictionHead, SpatialExactor2, TemporalExactorSTrans, PredictionHead2
from .stransformer import PatchEmbedding, PositionEmbedding, MHABlock, WindowMultiHeadAttention, UpsampleWithTransposedConv,SEResNet, PatchEmbedding2, PositionEmbedding2
import timm
from timm import create_model
from torchvision import transforms
import math
from .MSGNet import ScaleGraphModel2D
class PredictionHead2Head(nn.Module):
    def __init__(self, dims, hidden_dims=[256, 128], dropout=0.1, use_layer_norm=False):
        """
        MLP Head nhiều tầng với GELU activation.
        Args:
            dims (int): số chiều đầu vào (embedding size).
            hidden_dims (list[int]): danh sách số chiều cho các hidden layer.
            dropout (float): tỷ lệ dropout.
            use_layer_norm (bool): có dùng LayerNorm sau mỗi hidden layer không.
        """
        super().__init__()

        layers = []
        in_dim = dims
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU())  # thay ReLU bằng GELU
            layers.append(nn.Dropout(dropout))
            in_dim = hdim

        # Output layer: dự đoán 1 giá trị (regression hoặc logit)
        layers.append(nn.Linear(in_dim, 1))
        # layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, embedding):
        # embedding shape: [batch, H, W, dims]
        return self.layers(embedding)
class VIT_2Head(nn.Module):
    def __init__(self, config):
        super(VIT_2Head, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        

        self.dropout = config.MODEL.DROPOUT
        
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        
        
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        # self.upsample = SimpleUpsample(self.embed_dim * self.scale_time_factor, self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            ScaleGraphModel2D(config=self.config, d_model=config.MODEL.R, top_k=3, out_channel=self.embed_dim, L=config.MODEL.TIME_STEP) 
            # LSTMGridModel(self.config.DATA.HEIGHT_ESP, self.config.DATA.WIDTH_ESP, H_out=self.config.DATA.HEIGHT, W_out=self.config.DATA.HEIGHT, hidden_size=64, dropout=self.dropout, num_layers=2, out_channels=self.embed_dim, return_sequence=False)
            for _ in range(1)
        )

        print("Tích hợp LoRA vào khối spatial_encoder...")

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, drop_path_rate=self.dropout)
        # sd = torch.load("vit/vit_blocks_only.pth", map_location=self.config.DEVICE)

        
        
        vit.blocks = vit.blocks[:self.config.TRAIN.NUM_VITBLOCKS]
        
        lora_config = LoraConfig(
            r=self.config.MODEL.R,
            lora_alpha=self.config.MODEL.R,
            target_modules=["qkv", "fc1", "fc2", "proj"],
            lora_dropout=self.dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        vit.blocks = vit.blocks[:self.config.TRAIN.NUM_VITBLOCKS]
        
        peft_vit = get_peft_model(vit, lora_config)

        
        self.spatial_encoder = peft_vit.blocks 
        print("Tích hợp LoRA hoàn tất.")
        

        middle = self.embed_dim//2
        
        self.proj_x = nn.Sequential(
            nn.Linear(self.embed_dim, middle),
            nn.LayerNorm(middle),
            nn.Linear(middle, self.embed_dim)
        )
        self.proj_h = nn.Sequential(
            nn.Linear(self.embed_dim, middle),
            nn.LayerNorm(middle),
            nn.Linear(middle, self.embed_dim)
        )
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        # self.h_after = nn.Parameter(torch.zeros(self.config.TRAIN.BATCH_SIZE, self.config.DATA.HEIGHT, self.config.DATA.WIDTH, self.embed_dim))
        if self.prompt_type == 0:
            max_delta_t = config.MODEL.TEMPORAL.MAX_DELTA_T
            embed_dim = self.embed_dim
            pos_encoding = torch.zeros(max_delta_t, embed_dim)
            position = torch.arange(0, max_delta_t, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            if embed_dim % 2 == 1:
                pos_encoding[:, 1::2] = torch.cos(position * div_term)[:, :-1]
            else:
                pos_encoding[:, 1::2] = torch.cos(position * div_term)
            self.delta_t = nn.Parameter(pos_encoding, requires_grad=True)
        else:
            raise("Wrong prompt_type")
        self.predict_prob = PredictionHead2Head(
            self.embed_dim,
            use_layer_norm=config.MODEL.USE_LAYER_NORM,
            dropout=self.dropout
        )
        self.sigma1 = nn.Parameter(torch.zeros((), device=config.DEVICE))
        self.sigma2 = nn.Parameter(torch.zeros((), device=config.DEVICE))
        self.prediction_head = PredictionHead2(self.embed_dim,
                                                use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                                dropout=self.dropout)

    def cal_num_patches(self, img_size):
        
        t, h, w = img_size[0], img_size[1], img_size[2]
        pad_t = (self.patch_size - t % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_t, padded_h, padded_w = t + pad_t, h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size) * (padded_t // self.patch_size)
        return (padded_t // self.patch_size), num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    lt = int(lt)
                    lt -= 7
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt, 0)
                repetition_factors = (1, 1, 1, temporal_embedding.shape[3]//add_prompt.shape[3])
                # print(temporal_embedding.shape, add_prompt.shape)
                add_prompt = add_prompt.repeat(repetition_factors)
                
                # print(temporal_embedding.shape, add_prompt.shape)
                return temporal_embedding + add_prompt
            elif self.add_type == 1:
                for lt in lead_time:
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt, 0)
                repetition_factors = (1, 1, 1, self.scale_time_factor)
                add_prompt = add_prompt.repeat(repetition_factors)
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        if len(x) >= 3:
            esp = x[2]
        else: esp = None
        lead_time = x[1]
        x_begin = x[0]
        batch_size, n_ts, n_ft, h, w = x_begin.shape
        
        
        x = x_begin.permute(0, 2, 1, 3, 4)
        pad_t = (self.patch_size - n_ts % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        padded_t, padded_h, padded_w = n_ts + pad_t, h + pad_h, w + pad_w
        x = x.view(batch_size, n_ft, padded_t, padded_h, padded_w)
        x_sequence, x_grid = self.patch_embed(x)
        pos_embedding = self.pos_embed(x_grid)
        x = x_sequence + pos_embedding
        
        x = self.spatial_encoder(x)
        
        
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = x.reshape(batch_size, h_patch, w_patch, -1) 
        # print(x.shape)
        x = self.upsample(x)
        x = x[:, :h, :w, :]
        h_after = torch.zeros_like(x).to(self.config.DEVICE)
        lt = lead_time if torch.is_tensor(lead_time) else torch.as_tensor(lead_time, device=x.device)
        buckets = (lt-1)//100000
        unique_buckets = torch.unique(buckets)
        for b in unique_buckets.tolist():  
            idx = (buckets == b).nonzero(as_tuple=True)[0] 
            if idx.numel() == 0:
                continue
            esp_sub = esp.index_select(0, idx)
            lt_sub  = lt.index_select(0, idx)
            out_sub = self.esp_temporal[b//5](esp_sub, lt_sub) 
            
            h_after.index_copy_(0, idx, out_sub)
        
        # x = self.proj_x(x)
        x = self.add_prompt_vecs(x, lead_time)
        # h_after = self.proj_h(h_after)
        
        
        x = x + h_after
        x_prob = self.predict_prob(x)
        x = self.prediction_head(x)
        self.res = x
        x = x + x_begin[:, -1, -1:, :, :].permute(0, 2, 3, 1)
        # x = torch.where(x_prob > 0.7, x, x.new_full(x.shape, -1.0))
        return x, x_prob, self.sigma1, self.sigma2
