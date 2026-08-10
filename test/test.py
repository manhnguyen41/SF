import argparse
import torch
import wandb
import os
from src.utils import utils, get_scaler, test_func
from src.model import models
from src.model.baseline import cnn_lstm, conv_lstm
from src.utils.loss import ExpMagnitudeWeightedMAELoss, WeightedMSELoss, WeightedThresholdMSE
from src.utils.get_option import get_option
from src.utils.get_session_name import get_session_name
from src.utils.dataloader import CustomDataset

def load_checkpoint(model, checkpoint_path, device):
    """Load the model checkpoint from file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_dict'])
    return model

if __name__ == "__main__":
    # Get options and config
    args, config = get_option()
    config.WANDB.SESSION_NAME = get_session_name(config)

    # Set device
    if not torch.cuda.is_available():
        print("Device: CPU")
        device = torch.device("cpu")
    else:
        print("Device: GPU")
        device = torch.device("cuda")

    config.DEVICE = device

    # Preprocess data
    print("*************** Get scaler ***************")
    input_scaler, output_scaler = get_scaler.get_scaler(config)

    # Init dataset
    print("*************** Init dataset ***************")
    test_dataset = CustomDataset(mode='test', config=config, ecmwf_scaler=input_scaler, output_scaler=output_scaler)

    # Init model
    print("*************** Init model ***************")
    if config.MODEL.NAME == "model_v1":
        model = models.Model_Ver1(config)
        model = model.to(device)
    elif config.MODEL.NAME == "model_v2": 
        model = models.Model_Ver2(config)
        model = model.to(device)
    elif config.MODEL.NAME == "strans":
        model = models.SwinTransformer(config)
        model = model.to(device)
    elif config.MODEL.NAME == "strans-v2": 
        model = models.SwinTransformer_Ver2(config)
        model = model.to(device)
    elif config.MODEL.NAME == "cnn-lstm":
        model = cnn_lstm.CNN_LSTM(config)
        model = model.to(device)
    elif config.MODEL.NAME == "cnn-lstm-se":
        model = cnn_lstm.CNN_LSTM_SE(config)
        model = model.to(device)
    elif config.MODEL.NAME == "conv-lstm":
        model = conv_lstm.ConvLSTMModel(config)
        model = model.to(device)
    else:
        raise("Wrong model")
    
    model = model.to(device)

    # Load the model checkpoint
    checkpoint_path = f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt"
    model = load_checkpoint(model, checkpoint_path, device)
    print(f"Model loaded from {checkpoint_path}")

    # Set the loss function
    if config.LOSS.NAME == "mse":
        loss_func = torch.nn.MSELoss()
    elif config.LOSS.NAME == "expweightedloss":
        loss_func = ExpMagnitudeWeightedMAELoss(config.LOSS.k)
    elif config.LOSS.NAME == "weightedmse":
        loss_func = WeightedMSELoss(weight_func=config.LOSS.WEIGHT_FUNC)
    elif config.LOSS.NAME == "weightedthresholdmse":
        loss_func = WeightedThresholdMSE(high_weight=config.LOSS.HIGH_WEIGHT, low_weight=config.LOSS.LOW_WEIGHT, threshold=config.LOSS.GROUNDTRUTH_THRESHOLD)
    else:
        raise ValueError("Error: Not correct loss function name!")

    # Init wandb session
    if config.WANDB.STATUS:
        wandb.login(key='your_wandb_api_key')
        wandb.init(
            entity="aiotlab",
            project="SubSeasonalForecasting",
            group=config.WANDB.GROUP_NAME,
            name=f"{config.WANDB.SESSION_NAME}",
            config=config,
        )

    # Perform the testing
    test_func.test_func1(model, test_dataset, loss_func, config, output_scaler, device, "/mnt/disk1/tunn/Subseasonal_Prediction/fig")

    if config.WANDB.STATUS:
        wandb.finish()
