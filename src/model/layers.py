import numpy as np
import torch
import torch.nn as nn




class SpatialExactor2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, num_conv_layers=2, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.padding = (kernel_size - 1) // 2
        
        # Create list of conv layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        
        # First conv layer (in_channels -> out_channels)
        self.conv_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.padding
            )
        )
        if use_batch_norm:
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
        
        # Additional conv layers (out_channels -> out_channels)
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=self.padding
                )
            )
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(out_channels))
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # 1x1 conv for residual connection if input and output channels differ
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
    
    def forward(self, x):
        # x shape: [batch_size, window_length, n_channels, h, w]
        batch_size, window_length, n_channels, h, w = x.shape
        
        # Reshape to combine batch and window dimensions for conv2d
        x = x.reshape(-1, n_channels, h, w)
        
        # Save input for residual connection
        identity = x
        
        # Apply conv layers sequentially
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            # ReLU after every conv layer
            x = self.relu(x)
        
        # Apply residual connection
        if self.residual is not None:
            identity = self.residual(identity)
        x = x + identity
        
        # Final activation after residual connection
        x = self.relu(x)
        
        # Reshape back to original dimensions
        out_channels = x.size(1)
        x = x.view(batch_size, window_length, out_channels, h, w)
        
        return x
        

### Input shape 
class SpatialExactor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        # Calculate padding to maintain input dimensions
        # For a kernel size K, padding = (K-1)/2 to maintain same dimensions ok!
        self.padding = (kernel_size - 1) // 2
        
        # 1x1 convolution layer with padding
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )
        
        # BatchNorm
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x shape: [batch_size, window_length, n_channels, h, w]
        batch_size, window_length, n_channels, h, w = x.shape
        
        # Reshape to combine batch and window dimensions for conv2d
        # x shape: [batch_size * window_length, n_channels, h, w]
        x = x.reshape(-1, n_channels, h, w)
        
        # Apply convolution with padding
        x = self.conv(x)
        
        # Apply BatchNorm
        if self.use_batch_norm:
            x = self.bn(x)
        
        # Reshape back to original dimensions
        out_channels = x.size(1)
        
        x = x.view(batch_size, window_length, out_channels, h, w)
        
        return x
        


class Combined_Spatial(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], use_batch_norm=False):
        super().__init__()
        
        # Create multiple SpatialExactor instances with different kernel sizes
        self.exactors = nn.ModuleList([
            SpatialExactor(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                use_batch_norm=use_batch_norm,
            ) for k in kernel_sizes
        ])
        
    def forward(self, x):
        # Apply each exactor and stack results along a new dimension
        outputs = [exactor(x) for exactor in self.exactors]
        # Stack along new dimension after window_length
        # Shape: [batch_size, window_length, n_kernels * out_channels, h, w]
        return torch.concatenate(outputs, dim=2)

class TemporalExactor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer (unidirectional)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
    def forward(self, x):
        # x shape: [batch_size, window_length, channels, height, width]
        batch_size, window_length, channels, height, width = x.shape
        
        # Reshape for GRU: [batch_size * height * width, window_length, channels]
        x = x.permute(0, 3, 4, 1, 2)  # [batch, height, width, window_length, channels]
        x = x.reshape(-1, window_length, channels)
        
        # Apply GRU and get only the last hidden state
        hidden, _ = self.gru(x)
        hidden = torch.sum(hidden, dim=1)
        # [batch_size * height * width, hidden_size]
        
        # Reshape back: [batch_size, channels, height, width]
        output = hidden.reshape(batch_size, height, width, self.hidden_size)
        # output = output.permute(0, 3, 1, 2)
        
        return output
    
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # Thêm thông tin vị trí vào input cho Transformer
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TemporalExactorSTrans(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=4):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Transformer cần input_size = hidden_size, nên ta thêm 1 lớp linear nếu cần
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: [batch_size, n_ts, h_patch, w_patch, embed_dim]
        batch_size, n_ts, height, width, embed_dim = x.shape
        
        x = x.permute(0, 2, 3, 1, 4).reshape(-1, n_ts, embed_dim)
        
        # 1. Chiếu input về hidden_size
        x = self.input_proj(x)
        
        # 2. Thêm positional encoding
        x = self.pos_encoder(x)
        
        # 3. Đưa qua Transformer Encoder
        transformer_output = self.transformer_encoder(x)
        
        # 4. Tổng hợp output: Lấy trung bình của các output theo chiều thời gian
        # Đây là một cách phổ biến để có được một vector đại diện duy nhất
        aggregated_output = transformer_output.mean(dim=1) # Shape: [batch_size * h * w, hidden_size]
        
        # 5. Reshape lại
        output = aggregated_output.reshape(batch_size, height, width, self.hidden_size)
        
        return output
        


class PredictionHead(nn.Module):
    def __init__(self, dims, dropout=0, use_layer_norm=False, out_dim=1):
        super().__init__()
        t = dims//2
        # Two linear layers with ReLU activation in between
        if use_layer_norm:
            self.layers = nn.Sequential(
                nn.Linear(dims, t),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(t),
                nn.Linear(t, out_dim),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dims, t),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(t, out_dim)
            )
        
    def forward(self, embedding):
        # embedding shape: [batch_size, height, width, dims]
        # Apply linear layers to the last dimension
        return self.layers(embedding)
    
class PredictionHead2(nn.Module):
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

        self.layers = nn.Sequential(*layers)

    def forward(self, embedding):
        # embedding shape: [batch, H, W, dims]
        return self.layers(embedding)
import torch
import torch.nn as nn
import torch.fft


