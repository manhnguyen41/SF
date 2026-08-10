# Project Name

## Overview
This project is designed for [brief description of what the project does]. It includes configurable model options, training utilities, and dataset management features.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the project by updating the YAML configuration file.

## Usage
To run the project, execute the following command:
```bash
python main.py --cfg path/to/config.yaml
```

## Configuration
The configuration is managed using YAML files. You can specify various hyperparameters, model settings, and training configurations in the config file. Additionally, command-line arguments can override these configurations.

## Arguments
The script supports the following command-line arguments:

### General Options
- `--cfg` (str, required): Path to the configuration file.
- `--seed` (int): Random seed for reproducibility.
- `--output_norm` (flag): Enable output normalization.

### Architecture Options
#### Spatial Extractor
- `--spatial_type` (int): Type of spatial extractor.
- `--in_channel` (int): Number of input channels.
- `--spatial_out_channel` (int): Number of output channels.
- `--kernel_sizes` (list[int]): Kernel sizes for convolution layers.
- `--use_batch_norm` (flag): Enable batch normalization.
- `--num_cnn_layers` (int): Number of CNN layers.

#### Temporal Extractor
- `--temporal_hidden_size` (int): Hidden size for temporal GRU.
- `--temporal_num_layers` (int): Number of GRU layers.
- `--max_delta_t` (int): Maximum delta time.
- `--adding_type` (int): Type of addition (0 for addition, 1 for concatenation).
- `--prompt_type` (int): Type of prompt used.

#### Prediction Head
- `--use_layer_norm` (flag): Use layer normalization.
- `--dropout` (float): Dropout rate.

### Training Options
- `--batch_size` (int): Training batch size.
- `--window_length` (int): Temporal window length.
- `--num_epochs` (int): Number of training epochs.

### Learning Rate Scheduler
- `--use_lrscheduler` (flag): Enable learning rate scheduler.
- `--scheduler_type` (str): Type of scheduler (`CosineAnnealingLR`, `ReduceLROnPlateau`).
- `--cosine_t_max` (int): Number of epochs for cosine cycle.
- `--cosine_eta_min` (float): Minimum learning rate for cosine scheduler.
- `--plateau_mode` (str): Mode for ReduceLROnPlateau (`min` or `max`).
- `--plateau_factor` (float): Factor for reducing learning rate.
- `--plateau_patience` (int): Patience before reducing learning rate.
- `--plateau_min_lr` (float): Minimum learning rate.
- `--plateau_verbose` (flag): Enable verbose mode for scheduler.

### Data Loading
- `--data_idx_dir` (str): Directory for data indices.
- `--data_dir` (str): Directory for dataset.
- `--npyarr_dir` (str): Directory for numpy arrays.
- `--processed_ecmwf_dir` (str): Directory for processed ECMWF data.

### Loss Function
- `--loss_func` (str): Loss function (`mse`, `expweightedloss`, `weightedmse`, `weightedthresholdmse`).
- `--k` (float): Value used for `ExpMagnitudeWeightedLoss`.
- `--weight_func` (str): Weight function for weighted loss.
- `--high_weight` (float): High weight value.
- `--low_weight` (float): Low weight value.
- `--groundtruth_threshold` (int): Threshold for determining sample weight.

### Model Options
- `--model_type` (str): Type of model (`graph_only`, `default`).

### Early Stopping
- `--patience` (int): Patience for early stopping.
- `--checkpoint_dir` (str): Directory for saving checkpoints.
- `--delta` (float): Minimum change to qualify as an improvement.

### Optimizer
- `--optim_name` (str): Optimizer type (`adam`, `adamw`).
- `--lr` (float): Learning rate.
- `--l2_coef` (float): L2 regularization coefficient.
- `--epochs` (int): Number of epochs.

### WandB Integration
- `--group_name` (str): WandB group name.
- `--debug` (flag): Enable debug mode.

## Project Structure
```
project-root/
│── configs/                # Configuration files
│── data/                   # Dataset and processing scripts
│── models/                 # Model architectures
│── utils/                  # Utility scripts
│── main.py                 # Main entry point
│── README.md               # Documentation
```

## License
<!-- This project is licensed under [LICENSE NAME].
 -->
