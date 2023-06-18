import os
import os.path as osp
import numpy as np
from PIL import Image

import cv2
import pandas as pd
import torch
from torch.utils import data

import json
import random
import glob

from .augmentation import AugHeavy


class LaneDataset(data.Dataset):
    def __init__(self, 
                 root : str,  
                 imset : str,
                 to_crop: bool = True,
                 test: bool = False):
        self.root = root
        self.imset = imset
        self.data_dir = pd.read_csv(imset)
        self.aug = AugHeavy(to_crop)
        self.to_crop = to_crop
        self.test = test

    def __getitem__(self, 
                    index: int):
        data_row = self.data_dir.iloc[index]
        if (not self.test):
            frame_list = []
            for i in range(1, 6):
                img = data_row[f'img_{i}']
                try:
                    img = np.array(Image.open(img).convert('RGB'))
                except OSError:
                    print("OSError occured, using other image")
                    img = data_row[f'img_{i-1}'] if i > 1 else data_row[f'img_{i+1}']
                    img = np.array(Image.open(img).convert('RGB'))
                frame_list.append(img)
            else:
                img_height, img_width, _ = img.shape

            mask_list = []
            for i in range(1, 6):
                mask = data_row[f'label_{i}']
                mask = self.get_mask(mask, img_height, img_width, 5)
                mask_list.append(mask)

            frames, masks = self.aug(frame_list, mask_list)

            frames = np.array(frames) 
            masks = np.expand_dims(np.array(masks), axis=1)

            frames = torch.from_numpy(np.transpose(frames.copy(), (0, 3, 1, 2)).copy()).float() # (5, 3, 224, 224)
            masks = torch.from_numpy(masks).float() # (5, 1, 224, 224)

            return frames, masks
        else:
            max_attempts = 5
            for i in range(1, max_attempts + 1):
                try:
                    img = data_row[f'img_{i}']
                    img = np.array(Image.open(img).convert('RGB'))
                    break
                except OSError:
                    print(f"OSError occurred when loading image {i}, trying another image")
                    if i < max_attempts:
                        continue
            
            img_height, img_width, _ = img.shape
            patches = {}
            patches['img'] = (img_height, img_width)
            for i in range(img_height // 224):
                for j in range(img_width // 224):
                    frame_list = []
                    for k in range(1, 6):
                        img = data_row[f'img_{k}']
                        try:
                            img = np.array(Image.open(img).convert('RGB'))
                            img = img[i * 224:(i + 1) * 224, j * 224:(j + 1) * 224, :]
                        except OSError:
                            print("OSError occured, using other image")
                            img = data_row[f'img_{k - 1}'] if k > 1 else data_row[f'img_{k + 1}']
                            img = np.array(Image.open(img).convert('RGB'))
                            img = img[i*224:(i+1)*224, j*224:(j+1)*224, :]
                        frame_list.append(img)

                    mask_list = []
                    for k in range(1, 6):
                        mask = data_row[f'label_{k}']
                        mask = self.get_mask(mask, img_height, img_width, 5)
                        mask = mask[i*224:(i+1)*224, j*224:(j+1)*224]
                        mask_list.append(mask)

                    frames, masks = self.aug(frame_list, mask_list)

                    frames = np.array(frames) 
                    masks = np.expand_dims(np.array(masks), axis=1)

                    frames = torch.from_numpy(np.transpose(frames.copy(), (0, 3, 1, 2)).copy()).float() # (5, 3, 224, 224)
                    masks = torch.from_numpy(masks).float() # (5, 1, 224, 224)
                    patches[(i, j)] = [frames, masks]

            return patches


    def __len__(self) -> int:
        return len(self.data_dir)
    
    def get_mask(self,
             file_name: str,
             img_height: int,
             img_width: int,
             line_thickness: int):

        with open(file_name, 'r') as f:
            label = json.load(f)

        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for item in label["annotations"]:
            if "lane" in item["class"]:
                points = item["data"]
                points = np.array([(int(point['x']), int(point['y'])) for point in points], np.int32)
                cv2.polylines(mask, [points], isClosed=False, color=1, thickness=line_thickness)

        return mask
