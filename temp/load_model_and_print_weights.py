
import torch
import torch.nn as nn
from src.model.models import SwinTransformer_Ver4
from src.utils.get_option import get_option
# Ensure the SEResNet and SwinTransformer classes are imported correctly
# from your_model_file import SwinTransformer
args, config = get_option()
# Load the model from checkpoint
checkpoint_path = '/mnt/disk1/tunm/Subseasional_Forecasting/saved_checkpoints/data2-r1-test/checkpoint/data2-r1-test_Strans-V4_PS-4_Lr-5e-05_LF-mse_DR-0.55_LN-True-ST_0_3-ON_False_Seed-52_LRS-True_ReduceLROnPlateau-min-0.5-3.pt'
checkpoint = torch.load(checkpoint_path, weights_only=True)
def get_device():
    if torch.cuda.is_available():
        print("Device: GPU")
        return torch.device("cuda")
    else:
        print("Device: CPU")
        return torch.device("cpu")
device = get_device()
# Define your model (adjust the configuration as per your model definition)
model = SwinTransformer_Ver4(
    config=config
).to(device)
print(checkpoint.keys())
# Load model weights from the checkpoint
model.load_state_dict(checkpoint['model_dict'], strict=False)

# Function to print weights of a specific layer
def print_layer_weights(layer_name):
    for name, param in model.named_parameters():
        if layer_name in name and param.requires_grad:
            print(f"{name} - {param.shape}")
            print(param.data)

# Example: Print the weights of 'channel_attn' layer
print_layer_weights('channel_attn')

# Example: Print the weights of 'conv1' layer in SEResNet
print(model.channel_attn.conv1.weight)
