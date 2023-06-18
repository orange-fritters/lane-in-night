import os
from PIL import Image

def check_image_with_pil(image_path):
    try:
        Image.open(image_path).verify()
        return True
    except:
        return False

directory = '/home/mindong/lane-in-night/c_1920_1200_night_train_3'

for filename in os.listdir(directory):
    print(filename)
    if filename.endswith('.jpg') or filename.endswith('.jpg'):  # add more conditions if there are other image types
        file_path = os.path.join(directory, filename)
        result = check_image_with_pil(file_path)
        if not result:
            print(f'Image file {file_path} is not okay.')
        
