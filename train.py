from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import random_split, DataLoader

import numpy as np
import time
import argparse


from dataload.dataset import LaneDataset
from model.model import STM


def get_arguments():
    parser = argparse.ArgumentParser(description="LIN")
    parser.add_argument("-root", type=str, help="path to youtube-vos",default='data/train_f/data/1.Training')
    parser.add_argument("-imset", type=str, help="path to youtube-vos",default='image_paths.csv')
    parser.add_argument("-batch", type=int, help="batch size",default=4)
    parser.add_argument("-log_iter", type=int, help="log per x iters",default=500)
    parser.add_argument("-learning_rate", type=float, help="learning rate",default=1e-4)
    parser.add_argument("-epoch", type=int, help="epochs",default=20)

    return parser.parse_args()

def train_epoch(args, model, data_loader, optimizer, loss_type, epoch, device):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        frames, masks = data

        frames = frames.to(device)
        masks = masks.to(device)

        Estimates = torch.zeros_like(masks)
        Estimates[..., 0] = masks[..., 0]
        
        n0_key, n0_val = model("memorize", frames[..., 0], Estimates[..., 0])
        n1_logit = model("segment", frames[..., 1], n0_key, n0_val)
        n1_label = masks[..., 1]
        loss = loss_type(n1_logit, n1_label)

        Estimates[..., 1] = F.softmax(n1_logit, dim=1).detach()

        n1_key, n1_val = model("memorize", frames[..., 1], Estimates[..., 1])
        n2_logit = model("segment", frames[..., 2], n1_key, n1_val)
        n2_label = masks[..., 2]
        loss += loss_type(n2_logit, n2_label)

        Estimates[..., 2] = F.softmax(n2_logit, dim=1).detach()

        n2_key, n2_val = model("memorize", frames[..., 2], Estimates[..., 2])
        n3_logit = model("segment", frames[..., 3], n2_key, n2_val)
        n3_label = masks[..., 3]
        loss += loss_type(n3_logit, n3_label)

        Estimates[..., 3] = F.softmax(n3_logit, dim=1).detach()

        n3_key, n3_val = model("memorize", frames[..., 3], Estimates[..., 3])
        n4_logit = model("segment", frames[..., 4], n3_key, n3_val)
        n4_label = masks[..., 4]
        loss += loss_type(n4_logit, n4_label)

        Estimates[..., 4] = F.softmax(n4_logit, dim=1).detach()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.log_iter == 0:
            runtime = time.perf_counter() - start_iter
            left_sec = runtime / args.report_interval * (len_loader - iter)
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
    model.eval()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        frames, masks = data

        frames = frames.to(device)
        masks = masks.to(device)

        Estimates = torch.zeros_like(masks)
        Estimates[..., 0] = masks[..., 0]
        
        n0_key, n0_val = model("memorize", frames[..., 0], Estimates[..., 0])
        n1_logit = model("segment", frames[..., 1], n0_key, n0_val)
        n1_label = masks[..., 1]
        loss = loss_type(n1_logit, n1_label)

        Estimates[..., 1] = F.softmax(n1_logit, dim=1).detach()

        n1_key, n1_val = model("memorize", frames[..., 1], Estimates[..., 1])
        n2_logit = model("segment", frames[..., 2], n1_key, n1_val)
        n2_label = masks[..., 2]
        loss += loss_type(n2_logit, n2_label)

        Estimates[..., 2] = F.softmax(n2_logit, dim=1).detach()

        n2_key, n2_val = model("memorize", frames[..., 2], Estimates[..., 2])
        n3_logit = model("segment", frames[..., 3], n2_key, n2_val)
        n3_label = masks[..., 3]
        loss += loss_type(n3_logit, n3_label)

        Estimates[..., 3] = F.softmax(n3_logit, dim=1).detach()

        n3_key, n3_val = model("memorize", frames[..., 3], Estimates[..., 3])
        n4_logit = model("segment", frames[..., 4], n3_key, n3_val)
        n4_label = masks[..., 4]
        loss += loss_type(n4_logit, n4_label)

        Estimates[..., 4] = F.softmax(n4_logit, dim=1).detach()

        total_loss += loss.item()

        if iter % args.log_iter == 0:
            runtime = time.perf_counter() - start_iter
            left_sec = runtime / args.report_interval * (len_loader - iter)
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


def train(args):
    DATA_ROOT = args.root
    IMSET = args.imset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = LaneDataset(DATA_ROOT, IMSET)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation

# Split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch, 
                                num_workers=1,
                                shuffle = True, 
                                pin_memory=True)
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch,
                                num_workers=1,
                                pin_memory=True)
    
    model = STM()
    model.to(device=device)
    print("[] Model Loaded...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_type = nn.CrossEntropyLoss()

    start_epoch = 0
    print("[] Train start...")
    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(model, 
                                             train_loader, 
                                             optimizer, 
                                             loss_type, 
                                             epoch, 
                                             device, 
                                             args)
        val_loss, val_time = validate(model, 
                                      val_loader)

        print(
            f'Epoch=[{epoch + 1:2d}/{args.num_epochs:2d}] '
            f'TrainLoss={train_loss:.3f} '
            f'ValLoss={val_loss:.3f} '
            f'TrainTime={int(train_time)}s '
            f'ValTime={int(val_time)}s '
        )


if __name__ == '__main__':
    args = get_arguments()
    train(args)