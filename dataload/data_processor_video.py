import csv
import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="LIN")
    parser.add_argument("--root", type=str, help="path to data", default='data/lane_detected/Training/Raw/c_1280_720_night_train_1')
    parser.add_argument("--first", type=int, help="path to annotation", default=9954803)
    parser.add_argument("--last", type=int, help="path to annotation", default=9954866)
    parser.add_argument("--version", type=int, help="path to annotation", default=2)

    return parser.parse_args()

def process(args):
    raw_root = args.root
    labels_root = args.root.replace("Raw", "Label")
    min_images = 5

    # Prepare CSV
    with open(f'image_paths_vid_{args.version}.csv', 'w', newline='') as csvfile:
        fieldnames = ['folder', 'img_1', 'img_2', 'img_3', 'img_4', 'img_5', 
                    'label_folder', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(args.first, args.last, min_images - 1):  # Step by min_images - 1
            # Prepare paths for images
            img_paths = [os.path.join(raw_root, f"{j}.jpg") for j in range(i, i + min_images)]
            
            # Prepare paths for labels
            label_paths = [os.path.join(labels_root, f"{j}.json") for j in range(i, i + min_images)]

            # Write to CSV
            writer.writerow({
                'folder': raw_root,
                'img_1': img_paths[0], 'img_2': img_paths[1], 'img_3': img_paths[2], 
                'img_4': img_paths[3], 'img_5': img_paths[4],
                'label_folder': labels_root,
                'label_1': label_paths[0], 'label_2': label_paths[1], 
                'label_3': label_paths[2], 'label_4': label_paths[3], 
                'label_5': label_paths[4],
            })


if __name__ == '__main__':
    args = get_arguments()
    process(args)