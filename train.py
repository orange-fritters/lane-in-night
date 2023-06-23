from __future__ import division
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split

import os
import time
import argparse

from dataload.dataset import LaneDataset
from dataload.dataset_video import LaneDatasetVid
from model.model_h import STM


def get_arguments():
    parser = argparse.ArgumentParser(description="LIN")
    parser.add_argument("-root", type=str, help="path to data", default='data/train_f/data/1.Training')
    parser.add_argument("-imset", type=str, help="path to annotation", default='image_paths.csv')
    parser.add_argument("-batch", type=int, help="batch size", default=6)
    parser.add_argument("-log_iter", type=int, help="log per x iters", default=40)
    parser.add_argument("-learning_rate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-num_epochs", type=int, help="epochs", default=10)
    parser.add_argument("-num_workers", type=int, help="num workers", default=4)
    parser.add_argument("-save_dir", type=str, help="save directory", default='result/')
    parser.add_argument("-exp_name", type=str, help="experiment name", default='new')

    return parser.parse_args()

def train_epoch(args, model, data_loader, optimizer, loss_type, epoch, device):
    print("Training...")
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        frames, masks = data

        frames = frames.to(device)
        masks = masks.to(device)
        Estimates = torch.zeros_like(masks)
        Estimates[:, 0, ...] = masks[:, 0, ...]
        
        n0_key, n0_val = model("memorize", frames[:, 0, ...], Estimates[:, 0, ...])
        n1_logit = model("segment", frames[:, 1, ...], n0_key, n0_val)
        n1_label = masks[:, 1, ...]

        loss = loss_type(n1_logit, n1_label)

        Estimates[:, 1, ...] = torch.sigmoid(n1_logit).detach()

        n1_key, n1_val = model("memorize", frames[:, 1, ...], Estimates[:, 1, ...])
        n2_logit = model("segment", frames[:, 2, ...], n1_key, n1_val)
        n2_label = masks[:, 2, ...]
        loss += loss_type(n2_logit, n2_label)

        Estimates[:, 2, ...] = torch.sigmoid(n2_logit).detach()

        n2_key, n2_val = model("memorize", frames[:, 2, ...], Estimates[:, 2, ...])
        n3_logit = model("segment", frames[:, 3, ...], n2_key, n2_val)
        n3_label = masks[:, 3, ...]
        loss += loss_type(n3_logit, n3_label)

        Estimates[:, 3, ...] = torch.sigmoid(n3_logit).detach()

        n3_key, n3_val = model("memorize", frames[:, 3, ...], Estimates[:, 3, ...])
        n4_logit = model("segment", frames[:, 4, ...], n3_key, n3_val)
        n4_label = masks[:, 4, ...]
        loss += loss_type(n4_logit, n4_label)

        Estimates[:, 4, ...] = torch.sigmoid(n4_logit).detach()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.log_iter == 0:
            runtime = time.perf_counter() - start_iter
            left_sec = runtime / args.log_iter * (len_loader - iter)
            hour = left_sec // 3600
            minute = (left_sec - left_sec // 3600 * 3600) // 60
            print(
                f'Epoch=[{epoch + 1:2d}/{args.num_epochs:2d}] '
                f'Iter=[{iter:4d}/{len(data_loader):4d}] '
                f'Loss[Batch/Train]= {loss.item():.3f}/{total_loss / (iter + 1):3f} '
                f'Time= {int(runtime)}s '
                f'ETC={int(hour)}H {int(minute)}M '
            )
            start_iter = time.perf_counter()
        
        for module in model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()
    
    total_loss /= len_loader

    return total_loss, time.perf_counter() - start_epoch


def validate(model, data_loader, loss_type, epoch, device):
    print("Validating...")
    model.eval()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        frames, masks = data

        frames = frames.to(device)
        masks = masks.to(device)
        Estimates = torch.zeros_like(masks)
        Estimates[:, 0, ...] = masks[:, 0, ...]
        
        n0_key, n0_val = model("memorize", frames[:, 0, ...], Estimates[:, 0, ...])
        n1_logit = model("segment", frames[:, 1, ...], n0_key, n0_val)
        n1_label = masks[:, 1, ...]

        loss = loss_type(n1_logit, n1_label)

        Estimates[:, 1, ...] = torch.sigmoid(n1_logit).detach()

        n1_key, n1_val = model("memorize", frames[:, 1, ...], Estimates[:, 1, ...])
        n2_logit = model("segment", frames[:, 2, ...], n1_key, n1_val)
        n2_label = masks[:, 2, ...]
        loss += loss_type(n2_logit, n2_label)

        Estimates[:, 2, ...] = torch.sigmoid(n2_logit).detach()

        n2_key, n2_val = model("memorize", frames[:, 2, ...], Estimates[:, 2, ...])
        n3_logit = model("segment", frames[:, 3, ...], n2_key, n2_val)
        n3_label = masks[:, 3, ...]
        loss += loss_type(n3_logit, n3_label)

        Estimates[:, 3, ...] = torch.sigmoid(n3_logit).detach()

        n3_key, n3_val = model("memorize", frames[:, 3, ...], Estimates[:, 3, ...])
        n4_logit = model("segment", frames[:, 4, ...], n3_key, n3_val)
        n4_label = masks[:, 4, ...]
        loss += loss_type(n4_logit, n4_label)

        Estimates[:, 4, ...] = torch.sigmoid(n4_logit).detach()

        total_loss += loss.item()

        if iter % args.log_iter == 0:
            runtime = time.perf_counter() - start_iter
            left_sec = runtime / args.log_iter * (len_loader - iter)
            hour = left_sec // 3600
            minute = (left_sec - left_sec // 3600 * 3600) // 60
            print(
                f'Epoch=[{epoch + 1:2d}/{args.num_epochs:2d}] '
                f'Iter=[{iter:4d}/{len(data_loader):4d}] '
                f'Loss[Batch/Val]= {loss.item():.3f}/{total_loss / (iter + 1):3f} '
                f'Time= {int(runtime)}s '
                f'ETC={int(hour)}H {int(minute)}M '
            )
            start_iter = time.perf_counter()
        
        for module in model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()

    
    total_loss /= len_loader

    return total_loss, time.perf_counter() - start_epoch


def save_model(model, optimizer, epoch, loss, args):
    save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{epoch:03d}_{loss:.3f}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)


def create_data_loader(args,
                       dataset : torch.utils.data.Dataset, 
                       ver : str):
    
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation

    train_indices, val_indices = train_test_split(dataset_indices, 
                                                  test_size=val_size, 
                                                  train_size=train_size, 
                                                  shuffle=True)

    # Using these indices, create PyTorch datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Save indices to CSV files
    pd.Series(train_indices).to_csv(f"{args.exp_name}_train_indices_{ver}.csv")
    pd.Series(val_indices).to_csv(f"{args.exp_name}_val_indices_{ver}.csv")

    # Now, create your data loaders
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch,
                            num_workers=args.num_workers,
                            shuffle=True,
                            pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            num_workers=args.num_workers,
                            pin_memory=True)
    return train_loader, val_loader

def train(args):
    DATA_ROOT = args.root
    IMSET = args.imset

    IMSET1 = "image_paths.csv"
    IMSET2 = "image_paths_2.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = LaneDataset(DATA_ROOT, IMSET)
    dataset2 = LaneDatasetVid(DATA_ROOT, IMSET2)

    train_loader, val_loader = create_data_loader(args, dataset, "1")
    train_loader2, val_loader2 = create_data_loader(args, dataset2, "2")
    
    model = STM()
    model.to(device=device)
    print("[] Model Loaded...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_type = nn.BCEWithLogitsLoss()
    val_loss_type = nn.BCEWithLogitsLoss()

    start_epoch = 0
    print("[] Train start...")
    best_val_loss = 1000
    for epoch in range(start_epoch, args.num_epochs):
        train_loss1, train_time1 = train_epoch(args,
                                               model, 
                                               train_loader, 
                                               optimizer, 
                                               loss_type, 
                                               epoch, 
                                               device, 
                                             )
        train_loss2, train_time2 = train_epoch(args,
                                               model, 
                                               train_loader2, 
                                               optimizer, 
                                               loss_type, 
                                               epoch, 
                                               device, 
                                                )
        val_loss1, val_time1 = validate(model, 
                                        val_loader,
                                        val_loss_type,
                                        epoch,
                                        device)
        val_loss2, val_time2 = validate(model,
                                        val_loader2,
                                        val_loss_type,
                                        epoch,
                                        device)
        
        if val_loss1 + val_loss2 < best_val_loss:
            best_val_loss =val_loss1 + val_loss2
            print("New Best model ...")
            save_model(model, optimizer, epoch, val_loss1 + val_loss2, args)

        print(
            f'Epoch=[{epoch + 1:2d}/{args.num_epochs:2d}] '
            f'TrainLoss={train_loss1:.3f}, {train_loss2:.3f} '
            f'ValLoss={val_loss1:.3f}, {val_loss2:.3f} '
            f'TrainTime={int(train_time1 + train_time2)}s '
            f'ValTime={int(val_time1 + val_time2)}s '
        )


if __name__ == '__main__':
    args = get_arguments()
    train(args)