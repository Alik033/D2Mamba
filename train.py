from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time
import re
import math

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable

from options import opt, device
import dataset as dataset
from misc import *
from vgg import *
from model import D2Mamba
from ssim import *
from SWD import *

# ------------------------------
# Helpers
# ------------------------------
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def safe_check_tensor(x, name, ckpt_dir):
    """Check tensor for NaN/Inf and dump if invalid."""
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"[ERROR] Tensor {name} contains NaN/Inf. Dumping to {ckpt_dir}")
        # torch.save({name: x.detach().cpu()}, os.path.join(ckpt_dir, f"debug_bad_{name}.pt"))
        raise RuntimeError(f"Tensor {name} has NaN/Inf")

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    print("Underwater Image Enhancement")

    # Model
    netG = D2Mamba(base_dim=24, num_paths=4, mamba_layers=4).to(device)

    # Losses
    mse_loss = nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    ssim_loss = SSIMLoss(11)
    swd_loss_fn = SpectralWassersteinLoss(n_projections=64, use_lab=True)
    heuristic_loss_fn = nn.L1Loss()

    # Optimizer
    # optim_g = optim.AdamW(netG.parameters(), lr=5e-4, weight_decay=1e-4)
    optim_g = optim.AdamW(netG.parameters(), lr=1e-4, weight_decay=1e-4)


    # Dataset + Loader
    dataset_obj = dataset.Dataset_Load(
        data_path=opt.data_path,
        transform=dataset.ToTensor()
    )
    batches = int(dataset_obj.len / opt.batch_size)
    dataloader = DataLoader(dataset_obj, batch_size=opt.batch_size, shuffle=True)

    # Checkpoint directory
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    # Load checkpoint if available
    models_loaded = getLatestCheckpointName()
    latest_checkpoint_G = models_loaded

    if latest_checkpoint_G is None:
        start_epoch = 1
        print('No checkpoints found! Training from scratch.')
    else:
        checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G), weights_only=False)
        start_epoch = checkpoint_g['epoch'] + 1
        netG.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        for param_group in optim_g.param_groups:
            param_group['lr'] = opt.learning_rate_g
        print(f'Restoring model from checkpoint epoch {start_epoch}')

    # AMP support
    use_amp = False
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Enable anomaly detection (optional for debugging)
    enable_detect_anomaly = False
    if enable_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Training
    netG.train()

    for epoch in range(start_epoch, opt.end_epoch + 1):
        opt.total_mse_loss = 0.0
        opt.total_ssim_loss = 0.0
        opt.total_vgg_loss = 0.0
        opt.total_swd_loss = 0.0
        opt.total_G_loss = 0.0
        # opt.total_hu_loss = 0.0

        for i_batch, sample_batched in enumerate(dataloader):
            hazy_batch = sample_batched['hazy'].to(device)
            clean_batch = sample_batched['clean'].to(device)

            # Data sanity check
            safe_check_tensor(hazy_batch, "hazy_batch", opt.checkpoints_dir)
            safe_check_tensor(clean_batch, "clean_batch", opt.checkpoints_dir)
            

            optim_g.zero_grad()

            # Forward
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred_batch, _ = netG(hazy_batch)
            else:
                pred_batch, _ = netG(hazy_batch)

            # Clamp outputs (assuming input range [0,1])
            pred_batch = torch.clamp(pred_batch, 0.0, 1.0)

            # Losses
            batch_mse_loss = opt.lambda_mse * mse_loss(pred_batch, clean_batch)

            safe_pred = torch.clamp(pred_batch, 0.0, 1.0)
            safe_clean = torch.clamp(clean_batch, 0.0, 1.0)

            batch_ssim_loss = opt.lambda_ssim * ssim_loss(safe_pred, safe_clean)
            batch_swd_loss = opt.lambda_swd * swd_loss_fn(safe_pred)

            clean_vgg_feats = vgg(normalize_batch(safe_clean))
            pred_vgg_feats = vgg(normalize_batch(safe_pred))
            batch_vgg_loss = opt.lambda_vgg * mse_loss(
                pred_vgg_feats.relu2_2,
                clean_vgg_feats.relu2_2 + 1e-8
            )
            # batch_hu_loss = heuristic_loss_fn(h_preds, h_targets)


            # Check each component before summing
            safe_check_tensor(batch_mse_loss, "batch_mse_loss", opt.checkpoints_dir)
            safe_check_tensor(batch_ssim_loss, "batch_ssim_loss", opt.checkpoints_dir)
            safe_check_tensor(batch_swd_loss, "batch_swd_loss", opt.checkpoints_dir)
            safe_check_tensor(batch_vgg_loss, "batch_vgg_loss", opt.checkpoints_dir)

            total_loss = batch_mse_loss + batch_ssim_loss + batch_swd_loss + batch_vgg_loss 
            safe_check_tensor(total_loss, "total_loss", opt.checkpoints_dir)

            # total_loss = batch_mse_loss + batch_ssim_loss + batch_swd_loss + batch_vgg_loss

            # # Loss sanity check
            # safe_check_tensor(total_loss, "total_loss", opt.checkpoints_dir)

            # Backward
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optim_g)
                nn.utils.clip_grad_norm_(netG.parameters(), 1.0, norm_type=2)
                scaler.step(optim_g)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(netG.parameters(), 1.0, norm_type=2)
                optim_g.step()

            # Logging values
            opt.batch_mse_loss = float(batch_mse_loss.detach().cpu().item())
            opt.batch_vgg_loss = float(batch_vgg_loss.detach().cpu().item())
            opt.batch_ssim_loss = float(batch_ssim_loss.detach().cpu().item())
            opt.batch_swd_loss = float(batch_swd_loss.detach().cpu().item())

            opt.total_mse_loss += opt.batch_mse_loss
            opt.total_vgg_loss += opt.batch_vgg_loss
            opt.total_ssim_loss += opt.batch_ssim_loss
            opt.total_swd_loss += opt.batch_swd_loss

            opt.batch_G_loss = opt.batch_mse_loss + opt.batch_ssim_loss + opt.batch_vgg_loss + opt.batch_swd_loss
            opt.total_G_loss += opt.batch_G_loss

            print(
                f"\r Epoch : {epoch} | ({i_batch+1}/{batches}) | "
                f"mse: {opt.batch_mse_loss:.6f} | "
                f"ssim: {opt.batch_ssim_loss:.6f} | "
                f"vgg: {opt.batch_vgg_loss:.6f} | "
                f"swd: {opt.batch_swd_loss:.6f}",
                end='', flush=True
            )

        # Epoch summary
        print(
            f"\nFinished ep. {epoch}, lr = {get_lr(optim_g):.6f}, "
            f"total_mse = {opt.total_mse_loss:.6f}, "
            f"total_ssim = {opt.total_ssim_loss:.6f}, "
            f"total_vgg = {opt.total_vgg_loss:.6f}, "
            f"total_swd = {opt.total_swd_loss:.6f}"
        )

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optim_g.state_dict(),
            'MSE_loss': opt.total_mse_loss,
            'SSIM_loss': opt.total_ssim_loss,
            'VGG_loss': opt.total_vgg_loss,
            'SWD_loss': opt.total_swd_loss,
            'opt': opt,
            'total_loss': opt.total_G_loss
        }, os.path.join(opt.checkpoints_dir, f'netG_{epoch}.pt'))
