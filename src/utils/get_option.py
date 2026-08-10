import argparse
from yacs.config import CfgNode as CN
import yaml
from types import SimpleNamespace

def get_option():
    """
    Get model configurations using command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    
    parser = argparse.ArgumentParser(description='Model Configuration')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', required=True)
    
    # Global configs
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gsmap_time_step', type=int)
    parser.add_argument('--ecmwf_time_step', type=int)
    parser.add_argument('--output_norm', action='store_true')
    
    # Architecture options
    ##### Spatial Exactor options
    
    parser.add_argument('--spatial_type', type=int)
    
    parser.add_argument('--in_channel', type=int,
                        help='Number of input channels for spatial exactor')
    parser.add_argument('--spatial_out_channel', type=int, 
                        help='Number of output channels for spatial exactor')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5],
                        help='Kernel sizes for spatial exactor')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use batch normalization in spatial exactor')
    parser.add_argument('--num_cnn_layers', type=int, )
    
    # Temporal Exactor options
    parser.add_argument('--temporal_hidden_size', type=int, 
                        help='Hidden size for temporal GRU')
    parser.add_argument('--temporal_num_layers', type=int, 
                        help='Number of GRU layers')
    parser.add_argument('--max_delta_t', type=int)
    parser.add_argument('--adding_type', type=int, 
                        help='0 for addition, 1 for concatenation')
    
    parser.add_argument('--prompt_type', type=int, 
                        help='Type of prompt to use')
    
    # Prediction head options
    parser.add_argument('--use_layer_norm', action='store_true', 
                        help='Use layer normalization in prediction head')
    parser.add_argument('--dropout', type=float, 
                        help='Dropout rate for prediction head')
    
    # CNN-LSTM options
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for LSTM')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    
    # Training options
    parser.add_argument('--batch_size', type=int, 
                        help='Batch size for training')
    parser.add_argument('--window_length', type=int,
                        help='Temporal window length')

    parser.add_argument('--num_epochs', type=int,
                        help='Number of training epochs')
    
    parser.add_argument('--num_vit_blocks', type=int, 
                        help='Number of VIT blocks')
    ## Lr scheduler config
    parser.add_argument('--use_lrscheduler', action="store_true")
    parser.add_argument('--scheduler_type', type=str ,
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'], 
                        help='Type of scheduler to use')
    
    # CosineAnnealingLR arguments
    parser.add_argument('--cosine_t_max', type=int, 
                        help='Number of epochs for one cosine cycle (CosineAnnealingLR)')
    parser.add_argument('--cosine_eta_min', type=float, 
                        help='Minimum learning rate (CosineAnnealingLR)')
    
    # ReduceLROnPlateau arguments
    
    parser.add_argument('--plateau_mode', type=str, choices=['min', 'max'], 
                        help='Mode for ReduceLROnPlateau: minimize or maximize metric')
    parser.add_argument('--plateau_factor', type=float, 
                        help='Factor by which LR is reduced (ReduceLROnPlateau)')
    parser.add_argument('--plateau_patience', type=int, 
                        help='Number of epochs to wait before reducing LR (ReduceLROnPlateau)')
    parser.add_argument('--plateau_min_lr', type=float,
                        help='Minimum learning rate (ReduceLROnPlateau)')
    parser.add_argument('--plateau_verbose', action='store_true', 
                        help='Print LR updates (ReduceLROnPlateau)')
    
    # Dataloader options
    parser.add_argument('--data_idx_dir', type=str,
                        help='Directory to load data idx from')
    parser.add_argument('--esp_data_path', type=str,
                        help='Directory to load data esp from')
    parser.add_argument('--gauge_data_path', type=str,
                        help='Directory to load data from')
    parser.add_argument('--npyarr_dir', type=str,
                        help='Directory to load npyarr from')
    parser.add_argument('--processed_ecmwf_dir', type=str, 
                        help='Directory to store processed ecmwf data')
    parser.add_argument('--height', type=int,
                        help='Height of input data')
    parser.add_argument('--width', type=int,
                        help='Width of input data')
    parser.add_argument('--lat_start', type=float)
    parser.add_argument('--lon_start', type=float)
    parser.add_argument('--height_esp', type=int,
                        help='Height of input data')
    parser.add_argument('--width_esp', type=int,
                        help='Width of input data')
    parser.add_argument('--lat_esp_start', type=float)
    parser.add_argument('--lon_esp_start', type=float)
    
    ### LossFunction
    parser.add_argument('--loss_func', type=str)
    parser.add_argument('--k', type= float,  help="k value used for ExpMagnitudeWeightedLoss")
    parser.add_argument('--weight_func', type=str, help="only use this param when utilizing weighted mse loss")
    parser.add_argument("--high_weight", type=float,)
    parser.add_argument("--low_weight", type=float, )
    parser.add_argument('--groundtruth_threshold', type=int,  help="the threshold value to decide the weight of each sample")
    ### Dataloader
    parser.add_argument('--validate_every', type=int, )
    parser.add_argument("--num_workers",type=int, )
    
    ### model
    # parser.add_argument("--name", type=str, choices=["cnn-lstm", "model_v1", "conv-lstm", "strans"], default= "model_v1")
    parser.add_argument("--name", type=str, default= "model_v1")
    ### early stopping
    parser.add_argument("--patience", type=int)
    parser.add_argument("--checkpoint_dir",type=str, )
    parser.add_argument("--delta", type=float)
    parser.add_argument("--patch_size", type=int)
    ### optimizer
    parser.add_argument("--optim_name",type=str, choices=['adam','adamw'])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2_coef",type=float)
    parser.add_argument("--epochs", type=int)
    
    ### Wandb
    parser.add_argument("--group_name",type=str)
    parser.add_argument("--debug",action="store_true")
    
    #### loss config
    # parser.add_argument("--loss_func",type=str, default='mse', choices=['mse'])

    
    
    # Parse arguments
    args, unparsed = parser.parse_known_args()
    
    
    config = get_config(args)

    return args, config

def dict_to_namespace(d):
    """Recursively converts dictionary to SimpleNamespace."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)

def get_config(args):
    """Get configuration from YAML file with dot notation access."""
    with open(args.cfg, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace for dot notation access
    config = dict_to_namespace(config_dict)
    # Update config with command line args
    update_config(config, args)
    
    return config

def update_config(config, args):
    """Update config with command line arguments."""
    
    def _check_args(name):
        return hasattr(args, name) and getattr(args, name) is not None

    if _check_args("name"):
        config.MODEL.NAME = args.name
    if _check_args("seed"):
        config.MODEL.SEED = args.seed
    if _check_args("gsmap_time_step"):
        config.MODEL.TIME_STEP = args.gsmap_time_step
    if _check_args("ecmwf_time_step"):
        config.MODEL.ECMWF_TIME_STEP = args.ecmwf_time_step
    if _check_args('output_norm'):
        config.TRAIN.OUTPUT_NORM = args.output_norm
    if _check_args('spatial_type'):
        config.MODEL.SPATIAL.TYPE = args.spatial_type
    if _check_args('in_channel'):
        config.MODEL.IN_CHANNEL = args.in_channel
    if _check_args('spatial_out_channel'):
        config.MODEL.SPATIAL.OUT_CHANNEL = args.spatial_out_channel
    if _check_args('num_cnn_layers'):
        config.MODEL.SPATIAL.NUM_LAYERS = args.num_cnn_layers
    if _check_args('temporal_hidden_size'):
        config.MODEL.TEMPORAL.HIDDEN_DIM = args.temporal_hidden_size
    if _check_args('temporal_num_layers'):
        config.MODEL.TEMPORAL.NUM_LAYERS = args.temporal_num_layers
    if _check_args('max_delta_t'):
        config.MODEL.TEMPORAL.MAX_DELTA_T = args.max_delta_t
    if _check_args('adding_type'):
        config.MODEL.TEMPORAL.ADDING_TYPE = args.adding_type
    if _check_args('prompt_type'):
        config.MODEL.PROMPT_TYPE = args.prompt_type
    if _check_args('use_layer_norm'):
        config.MODEL.USE_LAYER_NORM = args.use_layer_norm
    if _check_args('dropout'):
        config.MODEL.DROPOUT = args.dropout
        

    if _check_args('batch_size'):
        config.TRAIN.BATCH_SIZE = args.batch_size
    if _check_args('window_length'):
        config.DATA.WINDOW_LENGTH = args.window_length
    if _check_args('num_epochs'):
        config.TRAIN.EPOCHS = args.num_epochs
    if _check_args('num_vit_blocks'):
        config.TRAIN.NUM_VITBLOCKS = args.num_vit_blocks
    if _check_args('lr'):
        config.OPTIMIZER.LR = args.lr
    if _check_args('use_lrscheduler'):
        config.LRS.USE_LRS = args.use_lrscheduler
    if _check_args('scheduler_type'):
        config.LRS.NAME = args.scheduler_type
    if _check_args('cosine_t_max'):
        config.LRS.COSINE_T_MAX = args.cosine_t_max
    if _check_args('cosine_eta_min'):
        config.LRS.COSINE_ETA_MIN = args.cosine_eta_min
    if _check_args('plateau_mode'):
        config.LRS.PLATEAU_MODE = args.plateau_mode
    if _check_args('plateau_factor'):
        config.LRS.PLATEAU_FACTOR = args.plateau_factor
    if _check_args('plateau_patience'):
        config.LRS.PLATEAU_PATIENCE = args.plateau_patience
    if _check_args('plateau_min_lr'):
        config.LRS.PLATEAU_MIN_LR = args.plateau_min_lr
    if _check_args('plateau_verbose'):
        config.LRS.PLATEAU_VERBOSE = args.plateau_verbose
    
    # Update loss function
    if _check_args('loss_func'):
        config.LOSS.NAME = args.loss_func
    if _check_args('loss_k'):
        config.LOSS.k = args.loss_k
    if _check_args('high_weight'):
        config.LOSS.HIGH_WEIGHT = args.high_weight
    if _check_args('low_weight'):
        config.LOSS.LOW_WEIGHT = args.low_weight
    if _check_args('groundtruth_threshold'):
        config.LOSS.GROUNDTRUTH_THRESHOLD = args.groundtruth_threshold
    if _check_args('weight_func'):
        config.LOSS.WEIGHT_FUNC = args.weight_func
    
    if _check_args('debug'):
        config.WANDB.STATUS = not args.debug
    if _check_args('kernel_sizes'):
        config.MODEL.SPATIAL.KERNEL_SIZES = args.kernel_sizes
    if _check_args('group_name'):
        config.WANDB.GROUP_NAME = args.group_name
    if _check_args('use_batch_norm'):
        config.MODEL.SPATIAL.USE_BATCH_NORM = args.use_batch_norm
    if _check_args('patch_size'):
        config.MODEL.PATCH_SIZE = args.patch_size
        
    if _check_args('data_idx_dir'):
        config.DATA.DATA_IDX_DIR =  args.data_idx_dir
    if _check_args('esp_data_path'):
        config.DATA.ESP_DATA_PATH = args.esp_data_path
    if _check_args('gauge_data_path'):
        config.DATA.GAUGE_DATA_PATH =  args.gauge_data_path
    if _check_args('npyarr_dir'):
        config.DATA.NPYARR_DIR =  args.npyarr_dir
    if _check_args('processed_ecmwf_dir'):
        config.DATA.PROCESSED_ECMWF_DIR =  args.processed_ecmwf_dir
    if _check_args('lat_start'):
        config.DATA.LAT_START =  args.lat_start
    if _check_args('lon_start'):
        config.DATA.LON_START =  args.lon_start
    if _check_args('height'):
        config.DATA.HEIGHT =  args.height
    if _check_args('width'):
        config.DATA.WIDTH =  args.width
        
    if _check_args('lat_esp_start'):
        config.DATA.LAT_ESP_START =  args.lat_esp_start
    if _check_args('lon_esp_start'):
        config.DATA.LON_ESP_START =  args.lon_esp_start
    if _check_args('height_esp'):
        config.DATA.HEIGHT_ESP =  args.height_esp
    if _check_args('width_esp'):
        config.DATA.WIDTH_ESP =  args.width_esp
    return config
