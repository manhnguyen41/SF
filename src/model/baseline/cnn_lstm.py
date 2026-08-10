import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Add channel Attention 
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SEResNet(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=2):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = ChannelSELayer(out_channels, reduction_ratio)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)  # Apply SE block here

        out += identity  # Residual connection (optional)
        out = self.relu(out)

        return out
    
import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Input:
      x is a list/tuple, we use x_begin = x[0]
      x_begin shape: [B, T, C, H, W]

    Output:
      y shape: [B, H, W, 1]
    """
    def __init__(self, config):
        super().__init__()

        in_ch = config.MODEL.IN_CHANNEL
        if in_ch is None:
            in_ch = 13

        hidden = config.MODEL.TEMPORAL.HIDDEN_DIM
        lstm_layers = getattr(config.MODEL.TEMPORAL, "NUM_LAYERS", 1)
        bidirectional = getattr(config.MODEL.TEMPORAL, "BIDIRECTIONAL", False)
        dropout = getattr(config.MODEL.TEMPORAL, "DROPOUT", 0.0)

        # CNN feature extractor (per-frame)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # LSTM nhận input theo từng pixel location qua time
        # Mỗi vị trí (h, w) có 1 chuỗi độ dài T, mỗi bước có feature dim = hidden
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden * 2 if bidirectional else hidden

        # Final projection to 1 channel
        self.head = nn.Conv2d(lstm_out_dim, 1, kernel_size=1)

    def forward(self, x):
        x_begin = x[0]  # [B, T, C, H, W]
        B, T, C, H, W = x_begin.shape

        # 1) CNN per frame
        xt = x_begin.view(B * T, C, H, W)          # [B*T, C, H, W]
        feat = self.cnn(xt)                        # [B*T, hidden, H, W]

        hidden = feat.shape[1]
        feat = feat.view(B, T, hidden, H, W)       # [B, T, hidden, H, W]

        # 2) Chuẩn bị cho LSTM
        # Với mỗi pixel (h, w), ta có chuỗi T vector feature dim=hidden
        # [B, T, hidden, H, W] -> [B, H, W, T, hidden]
        feat = feat.permute(0, 3, 4, 1, 2).contiguous()

        # Gộp B*H*W thành batch cho LSTM
        feat = feat.view(B * H * W, T, hidden)     # [B*H*W, T, hidden]

        # 3) LSTM theo time
        lstm_out, (h_n, c_n) = self.lstm(feat)     # lstm_out: [B*H*W, T, lstm_out_dim]

        # Lấy output ở time step cuối
        last_feat = lstm_out[:, -1, :]             # [B*H*W, lstm_out_dim]

        # 4) Khôi phục lại spatial map
        lstm_out_dim = last_feat.shape[-1]
        map_feat = last_feat.view(B, H, W, lstm_out_dim)   # [B, H, W, lstm_out_dim]
        map_feat = map_feat.permute(0, 3, 1, 2).contiguous()  # [B, lstm_out_dim, H, W]

        # 5) Head projection
        y = self.head(map_feat)                    # [B, 1, H, W]

        # Return [B, H, W, 1]
        y = y.permute(0, 2, 3, 1).contiguous()
        return y     
    

class CNN_LSTM_SE(nn.Module):
    def __init__(self, config):
        super(CNN_LSTM_SE, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Squeeze-and-Excitation Module
        self.channel_attn = SEResNet(in_channels=13, out_channels=13, reduction_ratio=2)  # Apply SEModule after conv2
        
        #LSTM layers
        # self.lstm_input_size = (64 * (args.height // 2) * (args.width // 2))  # After conv + pooling
        self.lstm_input_size = (64 * (17 // 2) * (17 // 2))  # After conv + pooling
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM, 
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=2, out_features=17 * 17)
        
        ### learnable params
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
    def add_prompt_vecs(self, out, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = out.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return out + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = out.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([out, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")    
        
    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        # Shape of x is (batch_size, n_t, n_f, h, w)
        batch_size, n_t, n_f, h, w = x.shape
        
        x = x.view(batch_size * n_t, n_f, h, w)
        x = self.channel_attn(x)
        x = x.reshape(batch_size, n_t, n_f, h, w)
        
        cnn_out = []
        for t in range(n_t):
            #Shape of x[:, t] is (batch_size, n_f, h, w)
            out = self.conv1(x[:, t]) # Shape: (batch_size, 32, h, w)
            out = F.relu(out)
            out = self.conv2(out)  # Shape: (batch_size, 64, h, w)
            out = F.relu(out)
            out = self.pool(out)  # Shape: (batch_size, 64, h//2, w//2)
            
            # Apply SEModule for feature recalibration
            out = self.se_module(out)  # Apply attention to features
            
            out = out.view(batch_size, -1)  # Flatten
            cnn_out.append(out)
            
        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, n_t, flattened_size)
        
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, n_t, hidden_dim)
        lstm_out = torch.sum(lstm_out, dim=1)  # Sum over n_t (batch_size, hidden_dim)
        
        out = self.fc(lstm_out)  # Shape: (batch_size, h*w)
        out = out.view(batch_size, h, w)  # Reshape to (batch_size, h, w)
        out = out.unsqueeze(-1)
        output = self.add_prompt_vecs(out, lead_time)
        
        return output