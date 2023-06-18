import os
import csv

labels_root_1 = "data/lane_detected/Training/Label/c_1280_720_night_train_1"
raw_root_1 = "data/lane_detected/Training/Raw/c_1280_720_night_train_1"

labels_root_2 = "data/lane_detected/Training/Label/c_1280_720_night_train_4"
raw_root_2 = "data/lane_detected/Training/Raw/c_1280_720_night_train_4"

labels_root_3 = "data/lane_detected/Training/Label/c_1920_1200_night_train_3"
raw_root_3 = "data/lane_detected/Training/Raw/c_1920_1200_night_train_3"

labels_root_4 = "data/lane_detected/Training/Label/c_1920_1200_night_train_4"
raw_root_4 = "data/lane_detected/Training/Raw/c_1920_1200_night_train_4"

labels = [labels_root_1, labels_root_2, labels_root_3, labels_root_4]
raw = [raw_root_1, raw_root_2, raw_root_3, raw_root_4]

min_images = 5

# Prepare CSV
with open('image_paths_2.csv', 'w', newline='') as csvfile:
    fieldnames = ['folder', 'img_1', 'img_2', 'img_3', 'img_4', 'img_5', 
                  'label_folder', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for labels_root, raw_root in zip(labels, raw):
        for root, dirs, files in sorted(os.walk(raw_root)):
            jpg_files = sorted([f for f in files if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
            batches = []
            current_batch = []

            for i in range(len(jpg_files) - 1):
                # if the current image and the next one are continuous
                if int(jpg_files[i+1].split('.')[0]) - int(jpg_files[i].split('.')[0]) == 1:
                    current_batch.append(jpg_files[i])
                    if len(current_batch) == min_images:
                        batches.append(current_batch)
                        current_batch = []
                else: # if not continuous, reset current batch
                    current_batch = []

            for batch in batches:
                # Prepare paths for images
                img_paths = [os.path.join(root, jpg_file) for jpg_file in batch]
                
                label_root = root.replace(raw_root, labels_root)
                label_paths = [os.path.join(label_root, jpg_file.replace('.jpg', '.json')) for jpg_file in batch]

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
