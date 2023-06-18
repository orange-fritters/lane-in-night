import cv2
import os
import argparse
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from cv2 import VideoWriter, VideoWriter_fourcc

from dataload.dataset_video import LaneDataset
from model.model import STM


def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss


def process_images(model, device, frames, masks):
    frames = frames.to(device)
    masks = masks.to(device)

    original_frames = []
    estimated_masks = []
    ground_truth = []

    Estimates = torch.zeros_like(masks)
    Estimates[:, 0, ...] = masks[:, 0, ...]
    n0_key, n0_val = model("memorize", frames[:, 0, ...], Estimates[:, 0, ...])
    n1_logit = model("segment", frames[:, 1, ...], n0_key, n0_val)
    n1_label = masks[:, 1, ...]

    original_frames.append(frames[:, 1, ...].cpu().numpy()[0])
    estimated_masks.append(torch.sigmoid(n1_logit).detach().cpu().numpy()[0, 0, ...])
    ground_truth.append(masks[:, 1, ...].cpu().numpy()[0, 0, ...])


    Estimates[:, 1, ...] = torch.sigmoid(n1_logit).detach()
    n1_key, n1_val = model("memorize", frames[:, 1, ...], Estimates[:, 1, ...])
    n2_logit = model("segment", frames[:, 2, ...], n1_key, n1_val)
    n2_label = masks[:, 2, ...]

    original_frames.append(frames[:, 2, ...].cpu().numpy()[0])
    estimated_masks.append(torch.sigmoid(n2_logit).detach().cpu().numpy()[0, 0, ...])
    ground_truth.append(masks[:, 2, ...].cpu().numpy()[0, 0,...])


    Estimates[:, 2, ...] = torch.sigmoid(n2_logit).detach()
    n2_key, n2_val = model("memorize", frames[:, 2, ...], Estimates[:, 2, ...])
    n3_logit = model("segment", frames[:, 3, ...], n2_key, n2_val)
    n3_label = masks[:, 3, ...]

    original_frames.append(frames[:, 3, ...].cpu().numpy()[0])
    estimated_masks.append(torch.sigmoid(n3_logit).detach().cpu().numpy()[0, 0, ...])
    ground_truth.append(masks[:, 3, ...].cpu().numpy() [0, 0, ...])


    Estimates[:, 3, ...] = torch.sigmoid(n3_logit).detach()
    n3_key, n3_val = model("memorize", frames[:, 3, ...], Estimates[:, 3, ...])
    n4_logit = model("segment", frames[:, 4, ...], n3_key, n3_val)
    n4_label = masks[:, 4, ...]

    original_frames.append(frames[:, 4, ...].cpu().numpy()[0])
    estimated_masks.append(torch.sigmoid(n4_logit).detach().cpu().numpy()[0, 0, ...])
    ground_truth.append(masks[:, 4, ...].cpu().numpy()[0, 0, ...])

    Estimates[:, 4, ...] = torch.sigmoid(n4_logit).detach()


    return original_frames, estimated_masks, ground_truth


def reconstruct_image(patches, canvas, img_type="original"):
    # type in ["original", "estimation", "ground_truth"]
    if img_type == "original":
        for (i, j), patch in patches.items():
            for k in range(4):
                canvas[k][:, i*224:(i+1)*224, j*224:(j+1)*224] = patch[k] # (3, 224, 224)
    elif img_type == "estimation":
        for (i, j), patch in patches.items():
            for k in range(4):
                patch[k] = ((patch[k] > 0.5) * 255).astype(np.uint8)
                canvas[k][i*224:(i+1)*224, j*224:(j+1)*224] = patch[k]
    else : # image type == ground_truth
        for (i, j), patch in patches.items():
            for k in range(4):
                patch[k] = (patch[k]* 255).astype(np.uint8)
                canvas[k][i*224:(i+1)*224, j*224:(j+1)*224] = patch[k]
    return canvas


def get_arguments():
    parser = argparse.ArgumentParser(description="LIN")
    parser.add_argument("-root", type=str, help="path to data", default='data/lane_detected/Training/Raw/c_1280_720_night_train_1')
    parser.add_argument("-csv", type=str, help="path to csv", default='image_paths_vid.csv')
    parser.add_argument("-model_pth", type=str, help="path to model pth", default='result/exp_3/006_0.550.pth')
    parser.add_argument("-video_name", type=str, help="video name", default='output_video.mp4')
    parser.add_argument("-version", type=int, help="video version", default=2)
    parser.add_argument("-img_height", type=int, help="img_height", default=1280)
    parser.add_argument("-img_width", type=int, help="img_height", default=720)

    return parser.parse_args()


def make_video(args):
    DATA_ROOT = args.root
    IMSET = args.csv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = LaneDataset(DATA_ROOT, IMSET, to_crop=False, test=True)

    test_loader = DataLoader(dataset,
                            batch_size=1, 
                            num_workers=4,
                            pin_memory=True)

    model = STM()
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model, _, epoch, loss = load_model(model, optimizer, args.model_pth)
    
    model.eval()
    
    save_dir = f'report/video_{args.version}'
    kernel = np.ones((15,15),np.uint8)
    FPS = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(f'output_video_{args.version}.mp4', fourcc, float(FPS), ((args.img_height // 224) * 224, (args.img_width // 224) * 224))
    j = 0
    for iter, data in enumerate(test_loader):
        original_frames, estimated_masks, ground_truth = {}, {}, {}
        img_height, img_width = data['img'] # assuming shape is [B, C, H, W]

        for key, patch in data.items():
            if key == 'img':
                continue
            patch_frames, patch_estimated_masks, patch_ground_truth = process_images(model, device, patch[0], patch[1])
            original_frames[key] = patch_frames
            estimated_masks[key] = patch_estimated_masks
            ground_truth[key] = patch_ground_truth

        rows, cols = img_height // 224, img_width // 224

        originals = [np.zeros((3, rows * 224, cols * 224), dtype=np.float32) for _ in range(4)]
        estimates = [np.zeros((rows * 224, cols * 224), dtype=np.uint8) for _ in range(4)]
        ground_truths = [np.zeros((rows * 224, cols * 224), dtype=np.uint8) for _ in range(4)]
        
        original_frames = reconstruct_image(original_frames, originals, img_type="original")
        estimated_masks = reconstruct_image(estimated_masks, estimates, img_type="estimation")
        ground_truth = reconstruct_image(ground_truth, ground_truths, img_type="ground_truth")

        for i in range(4):
            org_frame = original_frames[i].transpose(1, 2, 0)
            org_frame *= 255
            org_frame = org_frame.astype(np.uint8)
            org_frame = cv2.cvtColor(org_frame, cv2.COLOR_RGB2BGR)

                # Overlay the mask onto the original frame
            estimated_mask = cv2.dilate(estimated_masks[i], kernel, iterations = 1)

            mask = np.zeros_like(org_frame) # Create an empty mask with the same shape as org_frame
            mask[estimated_masks[i] == 255] = [0, 0, 255] # Assign red color to mask where estimated_mask has 255

            org_frame[estimated_mask == 255] = [0, 0, 255]
            
            # Save the image
            cv2.imwrite(f"{save_dir}/overlay_{j}.png", org_frame)
            j += 1

            # Write the overlay frame to the video
            video.write(org_frame)

        # Release the video file at the end
    video.release()

if __name__ == "__main__":
    args = get_arguments()
    make_video(args)
