import os
import torch
import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from src.utils.loss import get_station_from_grid
from src.utils import utils
import copy
import torch.nn.functional as F

def load_checkpoint(model, checkpoint_path, device):
    """Load the model checkpoint from file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_dict'])
    return model

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

def valid_func(model, valid_dataloader, early_stopping, loss_func, config, device):
    model.eval()
    with torch.no_grad():
        valid_epoch_loss = []
        for data in tqdm(valid_dataloader):
            input_data, lead_time, y_valid = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device)
            h = data.get("h", None)
            if h is not None:
                h = h.to(device)
                # h = h[:, :-1, :, :]
                y_ = model([input_data, lead_time, h])
            else:
                y_ = model([input_data, lead_time])
            
            y_ = get_station_from_grid(y_, y_valid, config) # (batch_size, num_station, 1)
            y_valid = y_valid[:,:,0] # (batch_size, num_station)
            loss = loss_func(y_.squeeze(), y_valid.squeeze())
            valid_epoch_loss.append(loss.item())
            del y_valid, y_, input_data, lead_time
            if h is not None:
                del h
        valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

    return valid_epoch_loss
def valid_func_2head(model, valid_dataloader, early_stopping, loss_func, config, device):
    model.eval()
    with torch.no_grad():
        valid_epoch_loss = []
        for data in tqdm(valid_dataloader):
            input_data, lead_time, y_valid, rain_prob = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['prob'].to(device)    
            h = data.get("h", None)
            
            if h is not None:
                h = h.to(device)
                # print(h.shape)
                # h = h[:, :-1, :, :]
                y_, y_prob, sigma1, sigma2 = model([input_data, lead_time, h])
            else:
                y_, y_prob, sigma1, sigma2 = model([input_data, lead_time])
            y_ = get_station_from_grid(y_, y_valid, config)
            y_prob = get_station_from_grid(y_prob, rain_prob, config)
            y_valid = y_valid[:,:,0]
            rain_prob = rain_prob[:,:,0]
            y_prob = y_prob[:, :, 0]
            pos_weight = torch.tensor([7.0], device=y_prob.device)
            loss1 = F.binary_cross_entropy_with_logits(y_prob, rain_prob.float(), pos_weight=pos_weight)
            loss = loss1
            # loss2 = loss_func(y_.squeeze(), y_valid.squeeze())
            # loss = 0.5*torch.exp(-2*sigma1)*loss1 \
            #         + 0.5*torch.exp(-2*sigma2)*loss2 \
            #         + (sigma1 + sigma2)
            
            valid_epoch_loss.append(loss.item())
            del y_valid, y_, input_data, lead_time
            if h is not None:
                del h
        valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

    return valid_epoch_loss
import gc
def valid_func_2head_phase_2(model, valid_dataloader, early_stopping, loss_func, config, device):
    model.eval()
    phase_1_path = config.MODEL.PHASE_1_PATH
    model_phase_1 = model_2head.VIT_2Head(config).to(device)
    model_phase_1.load_state_dict(torch.load(phase_1_path, map_location=torch.device('cpu'), weights_only=True)["best_dict"])
    model_phase_1.eval()
    for p in model_phase_1.parameters():
        p.requires_grad = False
    with torch.no_grad():
        valid_epoch_loss = []
        for data in tqdm(valid_dataloader):
            input_data, lead_time, y_valid, rain_prob = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['prob'].to(device)    
            h = data.get("h", None)
            # Phase 1 test
            if h is not None:
                h = h.to(device)
                # ----------------- DỰ BÁO MƯA/KO MƯA BẰNG MODEL PHASE 1 -----------------
                with torch.no_grad():
                    _, y_prob_phase1, _, _ = model_phase_1([input_data, lead_time, h])
            else:
                with torch.no_grad():
                    _, y_prob_phase1, _, _ = model_phase_1([input_data, lead_time])
                    
            y_prob_phase1 = get_station_from_grid(y_prob_phase1, rain_prob, config)  
            y_prob_phase1 = y_prob_phase1[:, :, 0]      
            y_prob_phase1 = torch.round(torch.sigmoid(y_prob_phase1))
            if h is not None:
                h = h.to(device)
                # print(h.shape)
                # h = h[:, :-1, :, :]
                y_, y_prob, sigma1, sigma2 = model([input_data, lead_time, h])
            else:
                y_, y_prob, sigma1, sigma2 = model([input_data, lead_time])
            y_ = get_station_from_grid(y_, y_valid, config)
            y_prob = get_station_from_grid(y_prob, rain_prob, config)
            y_valid = y_valid[:,:,0]
            rain_prob = rain_prob[:,:,0]
            y_prob = y_prob[:, :, 0]
            mask = (y_prob_phase1 == 1)
            # print(mask)
            if mask.any():
                loss1 = loss_func(y_[mask].squeeze(), y_valid[mask].squeeze())
            else:
                loss1 = torch.tensor(0.0, device=device)
            loss = loss1
            # loss2 = loss_func(y_.squeeze(), y_valid.squeeze())
            # loss = 0.5*torch.exp(-2*sigma1)*loss1 \
            #         + 0.5*torch.exp(-2*sigma2)*loss2 \
            #         + (sigma1 + sigma2)
            
            valid_epoch_loss.append(loss.item())
            del y_valid, y_, input_data, lead_time
            if h is not None:
                del h
        valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

    return valid_epoch_loss

def train_func(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, config, device):
    model.to(device)
    checkpoint_path = f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt"
    start_epoch = 0
    best_valid_loss = float('inf')
    best_model_state = None
    results = {'train_losses': [], 'valid_losses': [], 'learning_rates': []}

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}, loading model...")
        model = load_checkpoint(model, checkpoint_path, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and config.LRS.USE_LRS:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float('inf'))
        results = checkpoint.get("results", results)
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_valid_loss:.4f}")

    # Scheduler setup
    scheduler = None
    if config.LRS.USE_LRS:
        if config.LRS.NAME == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.LRS.COSINE_T_MAX,
                eta_min=config.LRS.COSINE_ETA_MIN
            )
        elif config.LRS.NAME == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config.LRS.PLATEAU_MODE,
                factor=config.LRS.PLATEAU_FACTOR,
                patience=config.LRS.PLATEAU_PATIENCE,
                min_lr=config.LRS.PLATEAU_MIN_LR,
                verbose=config.LRS.PLATEAU_VERBOSE
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.LRS.NAME}. Choose 'CosineAnnealingLR' or 'ReduceLROnPlateau'")

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    # Training loop
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            
            for i, data in enumerate(tqdm(train_dataloader)):
                
                optimizer.zero_grad()
                input_data, lead_time, y_train = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device)
                
                h = data.get("h", None)
                
                if h is not None:
                    h = h.to(device)
                    # print(h.shape)
                    # h = h[:, :-1, :, :]
                    y_ = model([input_data, lead_time, h])
                else:
                    y_ = model([input_data, lead_time])
                y_ = get_station_from_grid(y_, y_train, config)
                
                y_train = y_train[:,:,0]
                
                loss = loss_func(y_.squeeze(), y_train.squeeze())
                loss.backward()
                
                
                optimizer.step()
                
                epoch_loss.append(loss.item())
                # Free temps ASAP
                del y_train, y_, input_data, lead_time
                if h is not None:
                    del h
            
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            valid_epoch_loss =  valid_func(model, valid_dataloader, early_stopping, loss_func, config, device)

            # Scheduler step
            if config.LRS.USE_LRS and scheduler is not None:
                if config.LRS.NAME == 'CosineAnnealingLR':
                    scheduler.step()
                elif config.LRS.NAME == 'ReduceLROnPlateau':
                    scheduler.step(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)
            
            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                best_model_state = {k: v.to(device) for k, v in model.state_dict().items()}
                # Save checkpoint
            torch.save({
                    "model_dict": {k: v.to(device) for k, v in model.state_dict().items()},
                    "best_dict": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                    "results": results
                }, checkpoint_path)    

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{config.TRAIN.EPOCHS} | Train Loss: {train_epoch_loss:.4f} | "
                  f"Valid Loss: {valid_epoch_loss:.4f} | LR: {current_lr:.6f}")
            
            results['train_losses'].append(train_epoch_loss)
            results['valid_losses'].append(valid_epoch_loss)
            results['learning_rates'].append(current_lr)
            
            
            
            if config.WANDB.STATUS:
                wandb.log({
                    "loss/train_loss": train_epoch_loss,
                    "loss/valid_loss": valid_epoch_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    f"{config.LRS.NAME}/epoch": epoch if config.LRS.USE_LRS else "NoScheduler/epoch"
                })
        # =====================================================================
        # BẮT ĐẦU PHẦN DỌN DẸP BỘ NHỚ SAU MỖI EPOCH
        # =====================================================================
        torch.cuda.empty_cache() # Dọn dẹp bộ nhớ đệm không sử dụng trên GPU
        gc.collect()             # Buộc trình thu gom rác của Python chạy
        # =====================================================================
            

    return {
        'best_model_state': best_model_state,
        'final_train_loss': train_epoch_loss,
        'best_valid_loss': best_valid_loss,
        'train_losses': results['train_losses'],
        'valid_losses': results['valid_losses'],
        'learning_rates': results['learning_rates'],
        'scheduler_type': config.LRS.NAME if config.LRS.USE_LRS else 'None'
    }
    
import torch.nn.functional as F   
def train_func_unsupervised(model,
                            train_dataset,
                            valid_dataset,
                            early_stopping,
                            loss_func,            # unused (we use mse internally) but keep for API compatibility
                            optimizer,
                            config,
                            device,
                            mask_ratio=0.2):
    
    model.to(device)

    checkpoint_path = f"vit/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt"
    start_epoch = 0
    best_valid_loss = float('inf')
    best_model_state = None
    results = {'train_losses': [], 'valid_losses': [], 'learning_rates': []}

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        ck = torch.load(checkpoint_path, map_location=device)
        if "model_dict" in ck:
            model.load_state_dict(ck["model_dict"])
        else:
            model.load_state_dict(ck.get("model_state_dict", ck))
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        best_valid_loss = ck.get("best_valid_loss", best_valid_loss)
        results = ck.get("results", results)
        print(f"Resumed from epoch {start_epoch}, best_valid_loss={best_valid_loss:.6f}")
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) # Ensure path exists

    scheduler = None
    if config.LRS.USE_LRS:
        if config.LRS.NAME == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=config.LRS.COSINE_T_MAX, eta_min=config.LRS.COSINE_ETA_MIN)
        elif config.LRS.NAME == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode=config.LRS.PLATEAU_MODE,
                                          factor=config.LRS.PLATEAU_FACTOR, patience=config.LRS.PLATEAU_PATIENCE,
                                          min_lr=config.LRS.PLATEAU_MIN_LR, verbose=config.LRS.PLATEAU_VERBOSE)
        else:
            raise ValueError(f"Unsupported scheduler {config.LRS.NAME}")

    try:
        collate_fn = getattr(utils, 'custom_collate_fn')
    except AttributeError:
        collate_fn = None
        print("Warning: custom_collate_fn not found in utils, using default DataLoader collate_fn.")

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=collate_fn)

    def make_patch_mask(B, t, h, w, patch_size, mask_ratio, device):
        t_blocks = (t + (patch_size - t % patch_size) % patch_size) // patch_size
        h_blocks = (h + (patch_size - h % patch_size) % patch_size) // patch_size
        w_blocks = (w + (patch_size - w % patch_size) % patch_size) // patch_size
        N = t_blocks * h_blocks * w_blocks
        masks = torch.rand(B, N, device=device) < mask_ratio
        return masks, (t_blocks, h_blocks, w_blocks)

    def apply_mask_to_x(x_batch, masks, block_shape, patch_size):
        B, T, _F, H, W = x_batch.shape
        t_blocks, h_blocks, w_blocks = block_shape
        
        pad_t = (patch_size - T % patch_size) % patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        
        x_permuted = x_batch.permute(0, 2, 1, 3, 4) # (B, F, T, H, W)
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x_permuted, (0, pad_w, 0, pad_h, 0, pad_t))
        else:
            x_padded = x_permuted
        
        x_padded = x_padded.permute(0, 2, 1, 3, 4) # (B, T_pad, F, H_pad, W_pad)
        x_masked = x_padded.clone()
        
        idx = 0
        for ti in range(t_blocks):
            t0 = ti * patch_size; t1 = t0 + patch_size
            for hi in range(h_blocks):
                h0 = hi * patch_size; h1 = h0 + patch_size
                for wi in range(w_blocks):
                    w0 = wi * patch_size; w1 = w0 + patch_size
                    sel = masks[:, idx]
                    if sel.any():
                        x_masked[sel, t0:t1, :, h0:h1, w0:w1] = -1 # Zero out across all F channels
                    idx += 1
        return x_masked, (pad_t, pad_h, pad_w)

    # training loop
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            x_batch = batch['x'].to(device)         # (B, T, F, H, W)
            lead_time = batch['lead_time'].to(device)
            h = batch.get('h', None)
            if h is not None:
                h = h.to(device)

            B, T_orig, F_orig, H_orig, W_orig = x_batch.shape
            patch_size = model.patch_size

            masks, block_shape = make_patch_mask(B, T_orig, H_orig, W_orig, patch_size, mask_ratio, device)

            x_masked, pads = apply_mask_to_x(x_batch.clone().detach(), masks, block_shape, patch_size)
            
            optimizer.zero_grad()
            
            if h is not None:
                y_pred = model([x_masked, lead_time, h])   # (B, T, F, H, W)
            else:
                y_pred = model([x_masked, lead_time])     # (B, T, F, H, W)

            target_tensor = x_batch.to(device) # (B, T, F, H, W)


            if y_pred.shape != target_tensor.shape:
                y_pred = y_pred[:, :T_orig, :, :H_orig, :W_orig]
                if y_pred.shape != target_tensor.shape:
                    raise ValueError(f"After cropping, y_pred shape {y_pred.shape} still does not match target_tensor shape {target_tensor.shape}.")

            t_blocks, h_blocks, w_blocks = block_shape
            
            pixel_mask_5d = torch.zeros((B, T_orig, F_orig, H_orig, W_orig), dtype=torch.bool, device=device)
            idx = 0
            for ti in range(t_blocks):
                t0 = ti * patch_size; t1 = min(t0 + patch_size, T_orig) # Crop to original T
                for hi in range(h_blocks):
                    h0 = hi * patch_size; h1 = min(h0 + patch_size, H_orig)
                    for wi in range(w_blocks):
                        w0 = wi * patch_size; w1 = min(w0 + patch_size, W_orig)
                        sel = masks[:, idx]
                        if sel.any():
                            pixel_mask_5d[sel, t0:t1, :, h0:h1, w0:w1] = True
                        idx += 1

            masked_count = pixel_mask_5d.sum().item()
            if masked_count == 0:
                continue
            diff = (y_pred - target_tensor).masked_select(pixel_mask_5d)
            loss = (diff ** 2).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            del x_batch, x_masked, lead_time, y_pred
            if h is not None:
                del h
        train_epoch_loss = sum(train_losses) / max(1, len(train_losses))
        valid_epoch_loss = 0.0
        valid_steps = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Valid"):
                x_batch = batch['x'].to(device)
                lead_time = batch['lead_time'].to(device)
                h = batch.get('h', None)
                if h is not None: h = h.to(device)

                B, T_orig, F_orig, H_orig, W_orig = x_batch.shape
                masks, block_shape = make_patch_mask(B, T_orig, H_orig, W_orig, patch_size, mask_ratio, device)
                x_masked, pads = apply_mask_to_x(x_batch, masks, block_shape, patch_size)

                if h is not None:
                    y_pred = model([x_masked, lead_time, h])
                else:
                    y_pred = model([x_masked, lead_time])
                
                target_tensor = x_batch.to(device)

                if y_pred.shape != target_tensor.shape:
                    y_pred = y_pred[:, :T_orig, :, :H_orig, :W_orig]
                    if y_pred.shape != target_tensor.shape:
                        raise ValueError(f"Validation: After cropping, y_pred shape {y_pred.shape} still does not match target_tensor shape {target_tensor.shape}.")
                Bmask = masks
                pixel_mask_5d = torch.zeros((B, T_orig, F_orig, H_orig, W_orig), dtype=torch.bool, device=device)
                idx = 0
                for ti in range(t_blocks):
                    t0 = ti * patch_size; t1 = min(t0 + patch_size, T_orig)
                    for hi in range(h_blocks):
                        h0 = hi * patch_size; h1 = min(h0 + patch_size, H_orig)
                        for wi in range(w_blocks):
                            w0 = wi * patch_size; w1 = min(w0 + patch_size, W_orig)
                            sel = Bmask[:, idx]
                            if sel.any():
                                pixel_mask_5d[sel, t0:t1, :, h0:h1, w0:w1] = True
                            idx += 1

                masked_count = pixel_mask_5d.sum().item()
                if masked_count == 0:
                    continue
                
                diff = (y_pred - target_tensor).masked_select(pixel_mask_5d)
                valid_epoch_loss += (diff ** 2).mean().item()
                valid_steps += 1

        valid_epoch_loss = valid_epoch_loss / max(1, valid_steps)

        if config.LRS.USE_LRS and scheduler is not None:
            if config.LRS.NAME == 'CosineAnnealingLR':
                scheduler.step()
            else:
                scheduler.step(valid_epoch_loss)

        early_stopping(valid_epoch_loss, model)
        
        torch.save({
            "model_dict": model.state_dict(),
            "best_dict": best_model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "results": results
        }, checkpoint_path)

        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.vit_blocks.state_dict(), "vit/vit_blocks_only.pth") 
            torch.save({
                "model_dict": model.state_dict(),
                "best_dict": best_model_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_valid_loss": best_valid_loss,
                "results": results
            }, checkpoint_path.replace(".pt", "_best.pt"))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] train_loss={train_epoch_loss:.6f} valid_loss={valid_epoch_loss:.6f} lr={current_lr:.6f}")
        results['train_losses'].append(train_epoch_loss)
        results['valid_losses'].append(valid_epoch_loss)
        results['learning_rates'].append(current_lr)

        if getattr(config.WANDB, "STATUS", False) and wandb:
            wandb.log({
                "loss/train_loss": train_epoch_loss,
                "loss/valid_loss": valid_epoch_loss,
                "learning_rate": current_lr,
                "epoch": epoch
            })

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from training.")
        
    return {
        'best_model_state': best_model_state,
        'final_train_loss': train_epoch_loss,
        'best_valid_loss': best_valid_loss,
        'train_losses': results['train_losses'],
        'valid_losses': results['valid_losses'],
        'learning_rates': results['learning_rates']
    }  
    
from src.model import models, model_2head    
def train_func_2head(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, config, device):
    model.to(device)
    checkpoint_path = f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt"
    start_epoch = 0
    best_valid_loss = float('inf')
    best_model_state = None
    results = {'train_losses': [], 'valid_losses': [], 'learning_rates': []}

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}, loading model...")
        model = load_checkpoint(model, checkpoint_path, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and config.LRS.USE_LRS:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float('inf'))
        results = checkpoint.get("results", results)
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_valid_loss:.4f}")

    # Scheduler setup
    scheduler = None
    if config.LRS.USE_LRS:
        if config.LRS.NAME == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.LRS.COSINE_T_MAX,
                eta_min=config.LRS.COSINE_ETA_MIN
            )
        elif config.LRS.NAME == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config.LRS.PLATEAU_MODE,
                factor=config.LRS.PLATEAU_FACTOR,
                patience=config.LRS.PLATEAU_PATIENCE,
                min_lr=config.LRS.PLATEAU_MIN_LR,
                verbose=config.LRS.PLATEAU_VERBOSE
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.LRS.NAME}. Choose 'CosineAnnealingLR' or 'ReduceLROnPlateau'")

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    # Training loop
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            
            for i, data in enumerate(tqdm(train_dataloader)):
                # if i>=1: break
                optimizer.zero_grad()
                input_data, lead_time, y_train, rain_prob = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['prob'].to(device)
                
                h = data.get("h", None)
                
                if h is not None:
                    h = h.to(device)
                    # print(h.shape)
                    # h = h[:, :-1, :, :]
                    y_, y_prob, sigma1, sigma2 = model([input_data, lead_time, h])
                else:
                    y_, y_prob, sigma1, sigma2 = model([input_data, lead_time])
                y_ = get_station_from_grid(y_, y_train, config)
                y_prob = get_station_from_grid(y_prob, rain_prob, config)
                y_train = y_train[:,:,0]
                rain_prob = rain_prob[:,:,0]
                y_prob = y_prob[:, :, 0]
                pos_weight = torch.tensor([7.0], device=y_prob.device)
                loss1 = F.binary_cross_entropy_with_logits(y_prob, rain_prob.float(), pos_weight=pos_weight)
                loss = loss1
                # loss2 = loss_func(y_.squeeze(), y_train.squeeze())
                # loss = 0.5*torch.exp(-2*sigma1)*loss1 \
                #     + 0.5*torch.exp(-2*sigma2)*loss2 \
                #     + (sigma1 + sigma2)
                loss.backward()
                
                
                optimizer.step()
                
                epoch_loss.append(loss.item())
                # Free temps ASAP
                del y_train, y_, input_data, lead_time
                if h is not None:
                    del h
            
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            valid_epoch_loss =   valid_func_2head(model, valid_dataloader, early_stopping, loss_func, config, device)

            # Scheduler step
            if config.LRS.USE_LRS and scheduler is not None:
                if config.LRS.NAME == 'CosineAnnealingLR':
                    scheduler.step()
                elif config.LRS.NAME == 'ReduceLROnPlateau':
                    scheduler.step(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)
            
            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                best_model_state = {k: v.to(device) for k, v in model.state_dict().items()}
                # Save checkpoint
            torch.save({
                    "model_dict": {k: v.to(device) for k, v in model.state_dict().items()},
                    "best_dict": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                    "results": results
                }, checkpoint_path)    

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{config.TRAIN.EPOCHS} | Train Loss: {train_epoch_loss:.4f} | "
                  f"Valid Loss: {valid_epoch_loss:.4f} | LR: {current_lr:.6f}")
            
            results['train_losses'].append(train_epoch_loss)
            results['valid_losses'].append(valid_epoch_loss)
            results['learning_rates'].append(current_lr)
            
            
            
            if config.WANDB.STATUS:
                wandb.log({
                    "loss/train_loss": train_epoch_loss,
                    "loss/valid_loss": valid_epoch_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    f"{config.LRS.NAME}/epoch": epoch if config.LRS.USE_LRS else "NoScheduler/epoch"
                })
        # =====================================================================
        # BẮT ĐẦU PHẦN DỌN DẸP BỘ NHỚ SAU MỖI EPOCH
        # =====================================================================
        torch.cuda.empty_cache() # Dọn dẹp bộ nhớ đệm không sử dụng trên GPU
        gc.collect()             # Buộc trình thu gom rác của Python chạy
        # =====================================================================
            

    return {
        'best_model_state': best_model_state,
        'final_train_loss': train_epoch_loss,
        'best_valid_loss': best_valid_loss,
        'train_losses': results['train_losses'],
        'valid_losses': results['valid_losses'],
        'learning_rates': results['learning_rates'],
        'scheduler_type': config.LRS.NAME if config.LRS.USE_LRS else 'None'
    }

def train_func_2head_phase_2(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, config, device):
    model.to(device)
    phase_1_path = config.MODEL.PHASE_1_PATH
    model_phase_1 = model_2head.VIT_2Head(config).to(device)
    model_phase_1.load_state_dict(torch.load(phase_1_path, map_location=torch.device('cpu'), weights_only=True)["best_dict"])
    model_phase_1.eval()
    for p in model_phase_1.parameters():
        p.requires_grad = False
    checkpoint_path = f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt"
    start_epoch = 0
    best_valid_loss = float('inf')
    best_model_state = None
    results = {'train_losses': [], 'valid_losses': [], 'learning_rates': []}

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}, loading model...")
        model = load_checkpoint(model, checkpoint_path, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and config.LRS.USE_LRS:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float('inf'))
        results = checkpoint.get("results", results)
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_valid_loss:.4f}")

    # Scheduler setup
    scheduler = None
    if config.LRS.USE_LRS:
        if config.LRS.NAME == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.LRS.COSINE_T_MAX,
                eta_min=config.LRS.COSINE_ETA_MIN
            )
        elif config.LRS.NAME == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config.LRS.PLATEAU_MODE,
                factor=config.LRS.PLATEAU_FACTOR,
                patience=config.LRS.PLATEAU_PATIENCE,
                min_lr=config.LRS.PLATEAU_MIN_LR,
                verbose=config.LRS.PLATEAU_VERBOSE
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.LRS.NAME}. Choose 'CosineAnnealingLR' or 'ReduceLROnPlateau'")

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    # Training loop
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            
            for i, data in enumerate(tqdm(train_dataloader)):
                # if i>=1: break
                optimizer.zero_grad()
                input_data, lead_time, y_train, rain_prob = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['prob'].to(device)
                
                h = data.get("h", None)
                # Phase 1 test
                if h is not None:
                    h = h.to(device)
                    # ----------------- DỰ BÁO MƯA/KO MƯA BẰNG MODEL PHASE 1 -----------------
                    with torch.no_grad():
                        _, y_prob_phase1, _, _ = model_phase_1([input_data, lead_time, h])
                else:
                    with torch.no_grad():
                        _, y_prob_phase1, _, _ = model_phase_1([input_data, lead_time])
                        
                y_prob_phase1 = get_station_from_grid(y_prob_phase1, rain_prob, config)  
                y_prob_phase1 = y_prob_phase1[:, :, 0]      
                y_prob_phase1 = torch.round(torch.sigmoid(y_prob_phase1))
                
                # Phase 2 train
                if h is not None:
                    h = h.to(device)
                    # print(h.shape)
                    # h = h[:, :-1, :, :]
                    y_, y_prob, sigma1, sigma2 = model([input_data, lead_time, h])
                else:
                    y_, y_prob, sigma1, sigma2 = model([input_data, lead_time])
                y_ = get_station_from_grid(y_, y_train, config)
                y_prob = get_station_from_grid(y_prob, rain_prob, config)
                
                y_train = y_train[:,:,0]
                
                rain_prob = rain_prob[:,:,0]
                y_prob = y_prob[:, :, 0]
                mask = (y_prob_phase1 == 1)
                
                if mask.any():
                    
                    loss1 = loss_func(y_[mask].squeeze(), y_train[mask].squeeze())
                else:
                    loss1 = torch.tensor(0.0, device=device)
                loss = loss1
                # loss2 = loss_func(y_.squeeze(), y_train.squeeze())
                # loss = 0.5*torch.exp(-2*sigma1)*loss1 \
                #     + 0.5*torch.exp(-2*sigma2)*loss2 \
                #     + (sigma1 + sigma2)
                loss.backward()
                
                
                optimizer.step()
                
                epoch_loss.append(loss.item())
                # Free temps ASAP
                del y_train, y_, input_data, lead_time
                if h is not None:
                    del h
            
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            valid_epoch_loss =   valid_func_2head_phase_2(model, valid_dataloader, early_stopping, loss_func, config, device)

            # Scheduler step
            if config.LRS.USE_LRS and scheduler is not None:
                if config.LRS.NAME == 'CosineAnnealingLR':
                    scheduler.step()
                elif config.LRS.NAME == 'ReduceLROnPlateau':
                    scheduler.step(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)
            
            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                best_model_state = {k: v.to(device) for k, v in model.state_dict().items()}
                # Save checkpoint
            torch.save({
                    "model_dict": {k: v.to(device) for k, v in model.state_dict().items()},
                    "best_dict": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                    "results": results
                }, checkpoint_path)    

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{config.TRAIN.EPOCHS} | Train Loss: {train_epoch_loss:.4f} | "
                  f"Valid Loss: {valid_epoch_loss:.4f} | LR: {current_lr:.6f}")
            
            results['train_losses'].append(train_epoch_loss)
            results['valid_losses'].append(valid_epoch_loss)
            results['learning_rates'].append(current_lr)
            
            
            
            if config.WANDB.STATUS:
                wandb.log({
                    "loss/train_loss": train_epoch_loss,
                    "loss/valid_loss": valid_epoch_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    f"{config.LRS.NAME}/epoch": epoch if config.LRS.USE_LRS else "NoScheduler/epoch"
                })
        # =====================================================================
        # BẮT ĐẦU PHẦN DỌN DẸP BỘ NHỚ SAU MỖI EPOCH
        # =====================================================================
        torch.cuda.empty_cache() # Dọn dẹp bộ nhớ đệm không sử dụng trên GPU
        gc.collect()             # Buộc trình thu gom rác của Python chạy
        # =====================================================================
            

    return {
        'best_model_state': best_model_state,
        'final_train_loss': train_epoch_loss,
        'best_valid_loss': best_valid_loss,
        'train_losses': results['train_losses'],
        'valid_losses': results['valid_losses'],
        'learning_rates': results['learning_rates'],
        'scheduler_type': config.LRS.NAME if config.LRS.USE_LRS else 'None'
    }
