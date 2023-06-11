import os
import csv

labels_root = "data/train_f/data/1.Training/label"
raw_root = 'data/train_f/data/1.Training/raw'
min_images = 5

# Prepare CSV
with open('image_paths.csv', 'w', newline='') as csvfile:
    fieldnames = ['folder', 'img_1', 'img_2', 'img_3', 'img_4', 'img_5', 
                  'label_folder', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Iterate over all sub-folders of root
    for root, dirs, files in sorted(os.walk(raw_root)):
        jpg_files = [f for f in files if f.endswith('.jpg')]
        jpg_files.sort() 
        if len(jpg_files) >= min_images:
            for i in range(0, len(jpg_files), min_images):
                # If there are not enough images for the final batch, skip it
                if len(jpg_files[i:i+min_images]) < min_images:
                    continue
                # Prepare paths for images
                img_paths = [os.path.join(root, jpg_files[j]) for j in range(i, i+min_images)]
                
                # Prepare paths for labels
                # label is .json file, so we need to replace the extension
                label_root = root.replace(raw_root, labels_root)
                label_paths = [os.path.join(label_root, jpg_files[j].replace('.jpg', '.json')) for j in range(i, i+min_images)]

                # Write to CSV
                writer.writerow({
                    'folder': root,
                    'img_1': img_paths[0], 'img_2': img_paths[1], 'img_3': img_paths[2], 
                    'img_4': img_paths[3], 'img_5': img_paths[4],
                    'label_folder': label_root,
                    'label_1': label_paths[0], 'label_2': label_paths[1], 
                    'label_3': label_paths[2], 'label_4': label_paths[3], 
                    'label_5': label_paths[4],
                })
