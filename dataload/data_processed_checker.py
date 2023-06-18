import csv
import os

def check_continuity(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        print(reader.fieldnames)
        for row in reader:
            for i in range(1, 6): # because we have 5 images in each row
                img_file = row[f'img_{i}']
                label_file = row[f'label_{i}']

                # We extract the file names and convert them to integers
                img_num = int(os.path.basename(img_file).split('.')[0])
                label_num = int(os.path.basename(label_file).split('.')[0])

                if i < 5: # To avoid index error in the last element
                    next_img_file = row[f'img_{i+1}']
                    next_label_file = row[f'label_{i+1}']

                    # We extract the next file names and convert them to integers
                    next_img_num = int(os.path.basename(next_img_file).split('.')[0])
                    next_label_num = int(os.path.basename(next_label_file).split('.')[0])

                    # Checking if the numbers are consecutive
                    if not ((next_img_num - img_num == 1) and (next_label_num - label_num == 1)):
                        print(f"Discontinuity in row: {row}")
                        print(f"Discontinuity between {img_file} and {next_img_file}")
                        print(f"Or between {label_file} and {next_label_file}")

# Run the function
if __name__ == "__main__":
    check_continuity('image_paths_2.csv')
