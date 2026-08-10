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
class Model_Ver1(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if config.MODEL.SPATIAL.TYPE == 0:
            self.spatial_exactor = Combined_Spatial(in_channels=config.MODEL.IN_CHANNEL,
                                                out_channels=config.MODEL.SPATIAL.OUT_CHANNEL,
                                                kernel_sizes=config.MODEL.SPATIAL.KERNEL_SIZES,
                                                use_batch_norm=config.MODEL.SPATIAL.USE_BATCH_NORM)
            self.temporal_exactor = TemporalExactor(input_size= len(config.MODEL.SPATIAL.KERNEL_SIZES) * config.MODEL.SPATIAL.OUT_CHANNEL,
                                                hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
        elif config.MODEL.SPATIAL.TYPE == 1:
            self.spatial_exactor = SpatialExactor2(in_channels=config.MODEL.IN_CHANNEL,
                                                   out_channels= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                   kernel_size= 3,
                                                   use_batch_norm= config.MODEL.SPATIAL.USE_BATCH_NORM,
                                                   num_conv_layers= config.MODEL.SPATIAL.NUM_LAYERS)
            self.temporal_exactor = TemporalExactor(input_size= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                    hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                    num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
            
    
        ### learnable params
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, config.MODEL.TEMPORAL.HIDDEN_DIM))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(config.MODEL.TEMPORAL.HIDDEN_DIM,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=config.MODEL.DROPOUT)
        
        # Initialize weights
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
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module)
        
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")
        
    def forward(self, x):
        """
        input shape: [batch size, n_ts, n_fts, 17, 17]
        """
        ncmwf = x[0]
        
        lead_time = x[1]
        spatial_embedding = self.spatial_exactor(ncmwf) ##[batch_size, window_length, n_kernels * out_channels, h, w]
        
        temporal_embedding = self.temporal_exactor(spatial_embedding) ## [batch_size, height, width, hidden_dim]
        # temporal_embedding =  
        output = self.add_prompt_vecs(temporal_embedding, lead_time) # B, H, W, D
        
        output = self.prediction_head(output)
        
        return output
    
class SwinTransformer(nn.Module):
    def __init__(self, config):
        super(SwinTransformer, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution

        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = x.view(batch_size * n_ts, h_patch, w_patch, self.embed_dim)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)

        # Step 4: Apply window-based multi-head attention
        x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x 
  
class Model_Ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if config.MODEL.SPATIAL.TYPE == 0:
            self.spatial_exactor = Combined_Spatial(in_channels=config.MODEL.IN_CHANNEL,
                                                out_channels=config.MODEL.SPATIAL.OUT_CHANNEL,
                                                kernel_sizes=config.MODEL.SPATIAL.KERNEL_SIZES,
                                                use_batch_norm=config.MODEL.SPATIAL.USE_BATCH_NORM)
            self.temporal_exactor = TemporalExactor(input_size= len(config.MODEL.SPATIAL.KERNEL_SIZES) * config.MODEL.SPATIAL.OUT_CHANNEL,
                                                hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
        elif config.MODEL.SPATIAL.TYPE == 1:
            self.spatial_exactor = SpatialExactor2(in_channels=config.MODEL.IN_CHANNEL,
                                                   out_channels= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                   kernel_size= 3,
                                                   use_batch_norm= config.MODEL.SPATIAL.USE_BATCH_NORM,
                                                   num_conv_layers= config.MODEL.SPATIAL.NUM_LAYERS)
            self.temporal_exactor = TemporalExactor(input_size= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                    hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                    num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
            
    
        ### learnable params
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, config.MODEL.TEMPORAL.HIDDEN_DIM))
        
        else:
            raise("Wrong prompt_type")
        
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        
        self.channel_attn2 = SEResNet(in_channels=192, out_channels=192, reduction_ratio=16)
        
        self.channel_attn3 = SEResNet(in_channels=128, out_channels=128, reduction_ratio=16)
        
        self.prediction_head = PredictionHead(config.MODEL.TEMPORAL.HIDDEN_DIM,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=config.MODEL.DROPOUT)
        
        # Initialize weights
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
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module)
        
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")
        
    def forward(self, x):
        """
        input shape: [batch size, n_ts, n_fts, 17, 17]
        """
        ncmwf = x[0]
        lead_time = x[1]
        
        batch_size, n_ts, n_ft, h, w = ncmwf.shape
        
        ncmwf = ncmwf.view(batch_size * n_ts, n_ft, h, w)
        ncmwf = self.channel_attn(ncmwf)
        ncmwf = ncmwf.reshape(batch_size, n_ts, n_ft, h, w)
        
        spatial_embedding = self.spatial_exactor(ncmwf) ##[batch_size, window_length, n_kernels * out_channels, h, w]
        
        # batch_size, window_length, n_out_channels, h, w = spatial_embedding.shape
        # X_reshaped = spatial_embedding.view(batch_size * window_length, n_out_channels, h, w)
        # X_reshaped = self.channel_attn2(X_reshaped)
        # X_reshaped = X_reshaped.view(batch_size, window_length, n_out_channels, h, w)
        
        temporal_embedding = self.temporal_exactor(spatial_embedding) ## [batch_size, height, width, hidden_dim]
        # temporal_embedding = self.temporal_exactor(X_reshaped) ## [batch_size, height, width, hidden_dim]
        
        # temporal_embedding =  
        output = self.add_prompt_vecs(temporal_embedding, lead_time) # B, H, W, D
        
        output = self.prediction_head(output)
        
        return output
    
class SwinTransformer_Ver2(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver2, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution
        
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape

        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)
        
        # print(x.shape, "--------------------")
        x = self.channel_attn(x)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = x.view(batch_size * n_ts, h_patch, w_patch, self.embed_dim)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
       
        # Step 4: Apply window-based multi-head attention
        x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x
    
class SwinTransformer_Ver3(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver3, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        # self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        # self.embed_dim = 768
        self.embed_dim = 192
        # self.embed_dim = 384
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        
        
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution
        
        # vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, drop_rate = self.dropout, attn_drop_rate=self.dropout/2, drop_path_rate=self.dropout/2)
        self.spatial_encoder = vit.blocks[:config.TRAIN.NUM_VITBLOCKS]
        # for param in self.spatial_encoder.parameters():
        #         param.requires_grad = False
        # for blk in self.spatial_encoder[:3]:
        #     for param in blk.parameters():
        #         param.requires_grad = True

        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:    
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    # print(lt)
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        # Combine time and feature dimensions
        # sua thanh B, T, F, H, W
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim) ==> B, Patch, embeddim, use Con3D

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        # resize_transform = transforms.Resize((224, 224))
        # input_tensor_resized = resize_transform(x)
        x = self.spatial_encoder(x)
        # Step 4: Apply window-based multi-head attention
        # x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        

        
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim) ==> Khong can
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)
        self.res = x
        return x 
    
class SwinTransformer_Ver4(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver4, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        # self.channel_attention = nn.MultiheadAttention(config.DATA.HEIGHT*config.DATA.WIDTH*config.MODEL.ECMWF_TIME_STEP, config.DATA.HEIGHT, dropout=self.dropout)
        
        
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        
        
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        # self.upsample = SimpleUpsample(self.embed_dim * self.scale_time_factor, self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            # ScaleGraphModel2D(config=self.config, d_model=config.MODEL.R, top_k=3, out_channel=self.embed_dim, L=config.MODEL.TIME_STEP) 
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
        # for param in self.spatial_encoder.parameters():
        #     param.requires_grad = False
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
                    lt -= 1
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
        # x = x.reshape(batch_size, n_ft, n_ts * h * w)
        # x, _ = self.channel_attention(x, x, x)
        # x = x.reshape(batch_size, n_ft, n_ts, h, w)
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
        # h_after = torch.zeros_like(x).to(self.config.DEVICE)
        # lt = lead_time if torch.is_tensor(lead_time) else torch.as_tensor(lead_time, device=x.device)
        # buckets = (lt-1)//100000
        # unique_buckets = torch.unique(buckets)
        # for b in unique_buckets.tolist():  
        #     idx = (buckets == b).nonzero(as_tuple=True)[0] 
        #     if idx.numel() == 0:
        #         continue
        #     esp_sub = esp.index_select(0, idx)
        #     lt_sub  = lt.index_select(0, idx)
        #     out_sub = self.esp_temporal[0](esp_sub, lt_sub) 
            
        #     h_after.index_copy_(0, idx, out_sub)
        # h_after = self.esp_temporal[0](esp)
        # x = self.proj_x(x)
        x = self.add_prompt_vecs(x, lead_time)
        # h_after = self.proj_h(h_after)
        # print(x.shape, h_after.shape)
        
        x = x # + h_after
        x = self.prediction_head(x)
        self.res = x
        x = x # + x_begin[:, :, -1, :, :].sum(dim=1).unsqueeze(-1)
        return x 

class SwinTransformer_Ver4b(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver4b, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        # self.channel_attention = nn.MultiheadAttention(config.DATA.HEIGHT*config.DATA.WIDTH*config.MODEL.ECMWF_TIME_STEP, config.DATA.HEIGHT, dropout=self.dropout)
        
        
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        
        
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        # self.upsample = SimpleUpsample(self.embed_dim * self.scale_time_factor, self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            # ScaleGraphModel2D(config=self.config, d_model=config.MODEL.R, top_k=3, out_channel=self.embed_dim, L=config.MODEL.TIME_STEP) 
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
        # for param in self.spatial_encoder.parameters():
        #     param.requires_grad = False
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
                    lt -= 1
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
        # x = x.reshape(batch_size, n_ft, n_ts * h * w)
        # x, _ = self.channel_attention(x, x, x)
        # x = x.reshape(batch_size, n_ft, n_ts, h, w)
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
        
        h_after = self.esp_temporal[0](esp)
        # x = self.proj_x(x)
        # x = self.add_prompt_vecs(x, lead_time)
        # h_after = self.proj_h(h_after)
        # print(x.shape, h_after.shape)
        
        x = x + h_after
        x = self.prediction_head(x)
        self.res = x
        #print(x.shape, x_begin[:, :, -1, :, :].sum(dim=1).unsqueeze(-1).shape)
        x = x #+ x_begin[:, :, -1, :, :].sum(dim=1).unsqueeze(-1)
        return x 
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

import torch.nn as nn
import torch.nn.functional as F

class SimpleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(SimpleUpsample, self).__init__()
        self.scale_factor = scale_factor
        # Sử dụng 'bilinear' cho dữ liệu mượt, 'nearest' cho kết quả sắc nét hơn
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # Conv 1x1 để điều chỉnh số kênh và tinh chỉnh đặc trưng sau khi upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x có shape [B, H, W, C] từ các bước trước
        # Cần permute về [B, C, H, W] cho các lớp Conv2d và Upsample
        x = x.permute(0, 3, 1, 2) 
        x = self.upsample(x)
        x = self.conv(x)
        # Permute lại về định dạng [B, H, W, C] để phù hợp với phần còn lại của mô hình
        x = x.permute(0, 2, 3, 1)
        return x
from peft import LoraConfig, get_peft_model, TaskType
class SwinTransformer_Ver5(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver5, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        # self.channel_attention = nn.MultiheadAttention(config.DATA.HEIGHT*config.DATA.WIDTH*config.MODEL.ECMWF_TIME_STEP, config.DATA.HEIGHT, dropout=self.dropout)
        
        
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        
        
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        # self.upsample = SimpleUpsample(self.embed_dim * self.scale_time_factor, self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            # ScaleGraphModel2D(config=self.config, d_model=config.MODEL.R, top_k=3, out_channel=self.embed_dim, L=config.MODEL.TIME_STEP) 
            # LSTMGridModel(self.config.DATA.HEIGHT_ESP, self.config.DATA.WIDTH_ESP, H_out=self.config.DATA.HEIGHT, W_out=self.config.DATA.HEIGHT, hidden_size=64, dropout=self.dropout, num_layers=2, out_channels=self.embed_dim, return_sequence=False)
            for _ in range(1)
            
        )

        print("Tích hợp LoRA vào khối spatial_encoder...")

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, drop_path_rate=self.dropout) # True
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
        # for param in self.spatial_encoder.parameters():
        #     param.requires_grad = False
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
                    lt -= 1
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
        # x = x.reshape(batch_size, n_ft, n_ts * h * w)
        # x, _ = self.channel_attention(x, x, x)
        # x = x.reshape(batch_size, n_ft, n_ts, h, w)
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
        # h_after = torch.zeros_like(x).to(self.config.DEVICE)
        # lt = lead_time if torch.is_tensor(lead_time) else torch.as_tensor(lead_time, device=x.device)
        # buckets = (lt-1)//100000
        # unique_buckets = torch.unique(buckets)
        # for b in unique_buckets.tolist():  
        #     idx = (buckets == b).nonzero(as_tuple=True)[0] 
        #     if idx.numel() == 0:
        #         continue
        #     esp_sub = esp.index_select(0, idx)
        #     lt_sub  = lt.index_select(0, idx)
        #     out_sub = self.esp_temporal[0](esp_sub, lt_sub) 
            
        #     h_after.index_copy_(0, idx, out_sub)
        # h_after = self.esp_temporal[0](esp)
        # x = self.proj_x(x)
        x = self.add_prompt_vecs(x, lead_time)
        # h_after = self.proj_h(h_after)
        # print(x.shape, h_after.shape)
        
        x = x # + h_after
        x = self.prediction_head(x)
        self.res = x
        x = x # + x_begin[:, :, -1, :, :].sum(dim=1).unsqueeze(-1)
        return x 
######################################################################################

from src.model.unet import UNetDecoder32
from src.model.transformer import CrossAttnDecoder32
from src.model.gsmap_vit import VITGSMAP
class SwinTransformer_Ver6(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver6, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        # self.channel_attention = nn.MultiheadAttention(config.DATA.HEIGHT*config.DATA.WIDTH*config.MODEL.ECMWF_TIME_STEP, config.DATA.HEIGHT, dropout=self.dropout)
        
        
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        
        
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding2(embed_dim=self.embed_dim)
        
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        # self.upsample = SimpleUpsample(self.embed_dim * self.scale_time_factor, self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            # ScaleGraphModel2D(config=self.config, d_model=config.MODEL.R, top_k=3, out_channel=self.embed_dim, L=config.MODEL.TIME_STEP) 
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
        # for param in self.spatial_encoder.parameters():
        #     param.requires_grad = False
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
                    lt -= 1
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
        # x = x.reshape(batch_size, n_ft, n_ts * h * w)
        # x, _ = self.channel_attention(x, x, x)
        # x = x.reshape(batch_size, n_ft, n_ts, h, w)
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
        
        h_after = self.esp_temporal[0](esp)
        # x = self.proj_x(x)
        x = self.add_prompt_vecs(x, lead_time)
        # h_after = self.proj_h(h_after)
        # print(x.shape, h_after.shape)
        
        x = x + h_after
        x = self.prediction_head(x)
        self.res = x
        #print(x.shape, x_begin[:, :, -1, :, :].sum(dim=1).unsqueeze(-1).shape)
        x = x #+ x_begin[:, :, -1, :, :].sum(dim=1).unsqueeze(-1)
        return x 
#ver_5: not pretrain
#ver_4: num  vit
#ver_3: patch embedding

### round 1
#ver_6: full
#ver_5: not pretrain
#ver_4: not GsMAP
#ver_4b: not Lt embedding
#ver_3: spatial-temporal