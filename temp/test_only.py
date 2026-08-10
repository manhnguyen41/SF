import argparse

from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import wandb
import os

from src.utils.get_option import get_option
from src.model.models import Model_Ver1
from src.utils.dataloader import CustomDataset
from src.utils import utils, get_scaler, train_func, test_func
from src.model import models 
from src.utils.loss import ExpMagnitudeWeightedMAELoss, WeightedMSELoss, WeightedThresholdMSE
from src.utils.get_session_name import get_session_name
from src.model.baseline import cnn_lstm, conv_lstm

if __name__ == "__main__":
    args = get_option()
    
    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg)) 

    # Set session name:
    # args.session_name = f"SubseasonalModelV1_Lr-{args.lr}_LF-{args.loss_func}_DR-{args.dropout}_LN-{args.use_layer_norm}-ST_{args.spatial_type}_{args.num_cnn_layers}-ON_{args.output_norm}_Seed-{args.seed}"

    args.session_name = get_session_name(args)
    
    if args.loss_func == "mse":
        loss_func = nn.MSELoss()
    elif  args.loss_func == "expweightedloss":
        loss_func = ExpMagnitudeWeightedMAELoss(args.k)
    elif args.loss_func == "weightedmse":
        loss_func = WeightedMSELoss(weight_func=args.weight_func)
    elif args.loss_func == "weightedthresholdmse":
        loss_func = WeightedThresholdMSE(high_weight=args.high_weight, low_weight= args.low_weight, threshold= args.groundtruth_threshold)
    else:
        raise("Error: Not correct loss function name!")

    # Set device
    if not torch.cuda.is_available():
        print("Device: CPU")
        device = torch.device("cpu")
    else:
        print("Device: GPU")
        device = torch.device("cuda")
    
    
    # Set seed
    utils.seed_everything(args.seed)
    args.device = device
    
    
    
    # Preprocess data
    print("*************** Get scaler ***************")
    input_scaler, output_scaler = get_scaler.get_scaler(args)
    # breakpoint()
    #
    print("*************** Init dataset ***************")
    train_dataset = CustomDataset(mode='train', args=args, ecmwf_scaler=input_scaler, output_scaler= output_scaler)
    valid_dataset = CustomDataset(mode='valid', args=args, ecmwf_scaler=input_scaler, output_scaler= output_scaler)
    test_dataset = CustomDataset(mode='test', args=args, ecmwf_scaler=input_scaler, output_scaler= output_scaler)

    # Set training modules 
    #
    print("*************** Init training modules ***************")
    if not os.path.exists(f"saved_checkpoints/{args.group_name}/checkpoint/"):
        print(f"Make dir saved_checkpoints/{args.group_name}/checkpoint/ ...")
        os.makedirs(f"saved_checkpoints/{args.group_name}/checkpoint/")
    
    early_stopping = utils.EarlyStopping(
        patience = args.patience,
        verbose=True,
        delta=args.delta,
        path= f"saved_checkpoints/{args.group_name}/checkpoint/{args.session_name}.pt"
    )

    ## Init model 
    print("*************** Init model ***************")
    if args.model_type == "default":
        model = models.Model_Ver1(args)
        model = model.to(device)
    elif args.model_type == "cnn-lstm":
        model = cnn_lstm.CNN_LSTM(args)
        model = model.to(device)
    elif args.model_type == "conv-lstm":
        model = conv_lstm.ConvLSTMModel(args)
        model = model.to(device)
    else:
        raise("Wrong model")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    ## Set optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if  args.optim_name == "adam":
        optimizer = torch.optim.Adam(
        trainable_params, lr=args.lr, weight_decay=args.l2_coef
        )
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.l2_coef
        )
    else:
        raise("Error: Wrong optimizer name")
    
    # Init wandb session
    if not args.debug:
        wandb.login(key='ea35e57e56b147a6ec766aed718fb2479086f2fb')
        wandb.init(
            entity="aiotlab",
            project="SubSeasonalForecasting",
            group=args.group_name,
            name=f"testonly_{args.session_name}",
            config=config,
        )
        
    
    # print(f"\nFinal Results with {args.scheduler_type if args.use_lrscheduler else 'No Scheduler'}:")
    # print(f"Best Validation Loss: {results['best_valid_loss']:.4f}")
    # print(f"Final Train Loss: {results['final_train_loss']:.4f}")
    
    utils.load_model(model, f"saved_checkpoints/{args.group_name}/checkpoint/{args.session_name}.pt")
    test_func.test_func(model, test_dataset, loss_func, args, output_scaler, device)
