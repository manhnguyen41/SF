import torch
import torch.nn.functional as F
import torch.nn as nn

def get_station_from_grid(y_pred, y, config, step = 0.125):
    # y_pred: (batch_size, h, w, 1) 
    # y: (batch_size, num_station, 3) 
    batch_size = y.shape[0]
    num_station = y.shape[1]
    lat_start = config.DATA.LAT_START 
    lon_start = config.DATA.LON_START
    
    stations_lon = y[:, :, 1]  # (batch_size, num_station)
    stations_lat = y[:, :, 2]  # (batch_size, num_station)

    lat_idx = torch.floor((lat_start - stations_lat) / step).long()  # (batch_size, num_station)
    lon_idx = torch.floor((stations_lon - lon_start) / step).long()  # (batch_size, num_station)

    num_lat, num_lon = config.DATA.HEIGHT, config.DATA.WIDTH
    lat_idx = torch.clamp(lat_idx, 0, num_lat - 1)  # (batch_size, num_station)
    lon_idx = torch.clamp(lon_idx, 0, num_lon - 1)  # (batch_size, num_station)
    
    batch_idx = torch.arange(batch_size).view(-1, 1).expand(-1, num_station)  # (batch_size, num_station)
    
    pred_values = y_pred[batch_idx, lat_idx, lon_idx, 0]  # (batch_size, num_station)
    pred_values = pred_values.unsqueeze(-1)  # (batch_size, num_station, 1)
    
    return pred_values
def get_station_from_gsmap(y_pred, y, config, step = 0.125):
    # y_pred: (batch_size, h, w, 1) 
    # y: (batch_size, num_station, 3) 
    batch_size = y.shape[0]
    num_station = y.shape[1]
    lat_start = config.DATA.LAT_ESP_START 
    lon_start = config.DATA.LON_ESP_START
    
    stations_lon = y[:, :, 1]  # (batch_size, num_station)
    stations_lat = y[:, :, 2]  # (batch_size, num_station)

    lat_idx = torch.floor((lat_start - stations_lat) / step).long()  # (batch_size, num_station)
    lon_idx = torch.floor((stations_lon - lon_start) / step).long()  # (batch_size, num_station)

    num_lat, num_lon = config.DATA.HEIGHT_ESP, config.DATA.WIDTH_ESP
    lat_idx = torch.clamp(lat_idx, 0, num_lat - 1)  # (batch_size, num_station)
    lon_idx = torch.clamp(lon_idx, 0, num_lon - 1)  # (batch_size, num_station)
    
    batch_idx = torch.arange(batch_size).view(-1, 1).expand(-1, num_station)  # (batch_size, num_station)
    
    pred_values = y_pred[batch_idx, lat_idx, lon_idx, 0]  # (batch_size, num_station)
    pred_values = pred_values.unsqueeze(-1)  # (batch_size, num_station, 1)
    return pred_values
class MagnitudeWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(MagnitudeWeightedHuberLoss, self).__init__()
        self.delta = delta  # Threshold for Huber Loss

    def forward(self, y_pred, y_true):
        # Compute the error
        error = y_true - y_pred
        
        # Compute weights based on absolute target values
        weights = torch.abs(y_true)
        
        # Huber Loss calculation
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        huber_loss = 0.5 * quadratic**2 + self.delta * linear
        
        # Apply magnitude weighting
        weighted_loss = weights * huber_loss
        
        # Return mean loss
        return torch.mean(weighted_loss)

class MSLELoss(nn.Module):
    def forward(self, y_pred, y_true):
        # y_pred, y_true >= 0
        return torch.mean((torch.log1p(y_pred+1) - torch.log1p(y_true+1))**2)
class ExpMagnitudeWeightedMAELoss(nn.Module):
    def __init__(self, k=0.1, reduction='mean'):
        super(ExpMagnitudeWeightedMAELoss, self).__init__()
        self.k = k  # Scaling factor for exponential weighting
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, y_pred, y_true):
        # Expected shapes: y_pred, y_true = [batch, n_station]
        assert y_pred.shape == y_true.shape, "Prediction and target shapes must match"
        
        # Compute the absolute error
        error = torch.abs(y_true - y_pred)  # Shape: [batch, n_station]

        # Compute weights based on exponential of absolute target values
        weights = torch.exp(self.k * torch.abs(y_true))  # Shape: [batch, n_station]

        # Apply weighting to MAE
        weighted_loss = weights * error  # Shape: [batch, n_station]

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)  # Scalar: mean over batch and stations
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)  # Scalar: sum over batch and stations
        elif self.reduction == 'none':
            return weighted_loss  # Shape: [batch, n_station]
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_func='square', reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.weight_func = weight_func  # 'abs', 'square', or custom callable
        self.reduction = reduction      # 'mean', 'sum', or 'none'

    def forward(self, y_pred, y_true):
        # Expected shapes: y_pred, y_true = [batch, n_station]
        assert y_pred.shape == y_true.shape, "Prediction and target shapes must match"
        
        # Compute the squared error
        error = torch.abs(y_true - y_pred)  # Shape: [batch, n_station]
        error = error * error
        
        # Compute weights based on target values
        if self.weight_func == 'abs':
            weights = torch.abs(y_true + 1.25)  # w_ij = |y_ij| # 1.5
        elif self.weight_func == 'square':
            weights = (y_true + 1.4) ** 2     # w_ij = y_ij^2 #6
        elif self.weight_func == 'logarit':
            weights = torch.log1p(2.5 + y_true)
        elif self.weight_func == 'u_shape':
            # C1 = 0.8
            # C2 = 0.6
            # alpha = 1.0
            # weight_for_zeros = C1 * torch.exp(-alpha * (y_true+1))
            # weight_for_peaks = C2 * torch.log1p((2.5 + y_true))
            weights = torch.maximum((y_true + 1.5) ** 4.5, 5.0 - 5.0 * (y_true+1.5))
            new_weight_value = torch.tensor(5.0, device=y_true.device, dtype=y_true.dtype) # 3.0
            weights = torch.where(y_true <= -0.99, new_weight_value, weights)
            
        elif callable(self.weight_func):
            weights = weights *  self.weight_func(y_true)  # Custom function
        else:
            raise ValueError("weight_func must be 'abs', 'square', or a callable")
        
        # Apply weighting to MSE
        weighted_loss = error * weights  # Shape: [batch, n_station]

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)  # Scalar: mean over batch and stations
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)   # Scalar: sum over batch and stations
        elif self.reduction == 'none':
            return weighted_loss              # Shape: [batch, n_station]
        
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")



class WeightedThresholdMSE(nn.Module):
    def __init__(self, high_weight=10, low_weight=1, threshold=200):
        super().__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
    
    
    def forward(self, y_pred, y_true):
        # Compute squared error
        squared_error = (y_true - y_pred) ** 2
        
        # Create weight mask based on threshold
        weights = torch.where(y_true > self.threshold, 
                            torch.tensor(self.high_weight, dtype=y_true.dtype, device=y_true.device), 
                            torch.tensor(self.low_weight, dtype=y_true.dtype, device=y_true.device))
        
        # Apply weights to squared errors
        weighted_error = weights * squared_error
        
        # Return mean loss
        return weighted_error.mean()
    
class LogMagnitudeWeightedHuberLoss(nn.Module):
    def __init__(self, delta=10.0, alpha=0.05, reduction='mean'):
        super(LogMagnitudeWeightedHuberLoss, self).__init__()
        self.delta = delta        # Ngưỡng phân chia giữa MSE và MAE
        self.alpha = alpha        # Điều chỉnh độ mạnh của weight theo lượng mưa
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Shape: [batch, num_station]
        assert y_pred.shape == y_true.shape, "Prediction and target shapes must match"

        # Error
        error = y_true - y_pred
        abs_error = torch.abs(error)

        # Huber Loss
        quadratic = torch.minimum(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        huber = 0.5 * quadratic ** 2 + self.delta * linear

        # Trọng số log(|y|): log(1 + α * |y|)
        weights = torch.log1p(self.alpha * torch.abs(y_true))

        # Áp dụng trọng số
        weighted_loss = weights * huber

        # Giảm chiều
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)
        elif self.reduction == 'none':
            return weighted_loss
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")

import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.8, alpha = 0.5): # Giam gamma, tang beta
        """
        Focal MSE Loss for regression with robustness to outliers.
        gamma: focal modulation strength
        beta: non-linearity exponent
        reduction: 'mean', 'sum', or 'none'
        eps: small value to ensure numerical stability
        """
        super(QuantileLoss, self).__init__() 
        self.quantiles = quantile
        if isinstance(self.quantiles, float):
            self.quantiles = [self.quantiles]
        self.quantiles = torch.tensor(self.quantiles, dtype=torch.float32)
        self.alpha = alpha
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Shape mismatch"
        self.quantiles = self.quantiles.to(y_pred.device)
        error = y_true - y_pred
        q = self.quantiles[0]
        loss = torch.max(q * error, (q - 1) * error) * self.alpha + torch.abs(y_true - y_pred) * (1-self.alpha)
        
        
        
        
        return loss.mean()
        
class CombinedWeightedLoss(nn.Module):
    def __init__(self, gamma = 1.0, beta = 1.0):
        super(CombinedWeightedLoss, self).__init__()
        # Trọng số có thể học được cho mỗi trạm
        self.log_vars = None
        self.gamma = gamma
        self.beta = beta
        

    def forward(self, y_pred, y_true):
        # 1. Tính loss động dựa trên y_true
        if self.log_vars is None:
            # Lấy số lượng trạm (num_stations) từ chiều cuối cùng của y_true
            num_stations = y_true.shape[-1]
            device = y_true.device # Lấy device của tensor đầu vào
            
            # Bây giờ chúng ta mới tạo nn.Parameter với kích thước chính xác
            # và đăng ký nó vào module.
            # Rất quan trọng: phải gán nó vào self.log_vars để PyTorch
            # nhận ra nó là một tham số của module.
            self.log_vars = nn.Parameter(torch.zeros(num_stations, device=device))
        error_sq = torch.abs(y_pred - y_true) ** 2
        #dynamic_weights = torch.maximum((y_true + 1.5) ** 3.7, 5.0 - 5.0 * (y_true+1.5))
        weights = torch.maximum((y_true + 1.5) ** 4.25, 5.0 - 5.0 * (y_true+1.5)) # 3.7
        new_weight_value = torch.tensor(3.0, device=y_true.device, dtype=y_true.dtype)
        weights = torch.where(y_true <= -0.99, new_weight_value, weights)
        dynamic_weighted_loss_per_sample = weights * error_sq # Shape: [B, num_stations]
        
        # 2. Lấy trung bình loss động cho mỗi trạm
        mean_loss_per_station = torch.mean(dynamic_weighted_loss_per_sample, dim=0) # Shape: [num_stations]
        
        # 3. Áp dụng Uncertainty Weighting
        loss_terms = 0.5 * torch.exp(-self.log_vars) * (mean_loss_per_station ** self.gamma) + 0.5 * self.log_vars * self.beta
        total_loss = torch.sum(loss_terms)
        
        return total_loss
import torch
import torch.nn as nn


