import random, os
import numpy as np
import torch
import wandb
import numpy as np

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], np.ndarray):
            data_array = np.stack([b[key] for b in batch], axis=0)
            collated_batch[key] = torch.from_numpy(data_array).float()
        elif isinstance(batch[0][key], float):
            collated_batch[key] = torch.tensor([b[key] for b in batch], dtype=torch.float32)
        elif isinstance(batch[0][key], int):
            collated_batch[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
        else:
            collated_batch[key] = [b[key] for b in batch] 
    return collated_batch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_checkpoint(model, path):
    checkpoints = {
        "model_dict": model.state_dict(),
    }
    torch.save(checkpoints, path)


def load_model(model, checkpoint_path):
    #model_dict best_dict
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["best_dict"])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=3, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score + self.delta > self.best_score:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        from pathlib import Path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        checkpoints = {"model_dict": model.state_dict()}
        torch.save(checkpoints, self.path)
        self.val_loss_min = val_loss


def to_float(x, device):
    if isinstance(x,list):
        list_x = []
        for x_i in x:
            x_i = x_i.to(device).float()
            list_x.append(x_i)
        x = list_x
    else:
        x = x.to(device).float()
        
    return x