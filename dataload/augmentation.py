import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import random
import cv2
from PIL import Image


class Flip(object):
    def __init__(self,rate):
        self.rate = rate
    def __call__(self,images,labels):
        if random.random() < self.rate:
            for i in range(5):
                # Random flipping
                images[i] = np.fliplr(images[i]).copy()  # HWC
                labels[i] = np.fliplr(labels[i]).copy()  # HW
        return images,labels


class RandomSizedCrop(object):
    def __init__(self, scale, crop_size):
        self.scale = scale
        self.crop_size = crop_size

    def __call__(self, images, labels):
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        for i in range(5): 
            h, w = labels[i].shape
            h, w = (max(224, int(h * scale_factor)), max(224, int(w * scale_factor)))
            images[i] = (cv2.resize(images[i], (w, h), interpolation=cv2.INTER_LINEAR))
            labels[i] = Image.fromarray(labels[i]).resize((w, h), resample=Image.NEAREST)
            labels[i] = np.asarray(labels[i], dtype=np.int8)
        ob_loc = ((labels[0] + labels[1] + labels[2]) > 0).astype(np.uint8)
        box = cv2.boundingRect(ob_loc)

        x_min = box[0]
        x_max = box[0] + box[2]
        y_min = box[1]
        y_max = box[1] + box[3]

        if x_max - x_min > 224:
            start_w = random.randint(x_min, x_max - 224)
        elif x_max - x_min == 224:
            start_w = x_min
        else:
            start_w = random.randint(max(0, x_max - 224), min(x_min, w - 224))

        if y_max - y_min > 224:
            start_h = random.randint(y_min, y_max - 224)
        elif y_max - y_min == 224:
            start_h = y_min
        else:
            start_h = random.randint(max(0, y_max - 224), min(y_min, h - 224))

        end_h = start_h + 224
        end_w = start_w + 224
        for i in range(5):
            start_h = random.randint(start_h - 20, start_h + 20)
            start_h = max(0, start_h)
            start_h = min(h - 224, start_h)
            start_w = random.randint(start_w - 20, start_w + 20)
            start_w = max(0, start_w)
            start_w = min(w - 224, start_w)
            end_h = start_h + 224
            end_w = start_w + 224
            images[i] = images[i][start_h:end_h, start_w:end_w]/255.
            labels[i] = labels[i][start_h:start_h + 224, start_w:start_w + 224]

        return images, labels
    

class AugHeavy(object):
    def __init__(self, to_crop=True):
        self.affinity = iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.Affine(rotate=(-30, 30))
            ),
            iaa.Sometimes(
                0.5,
                iaa.Affine(shear=(-15, 15))
            ),
            iaa.Sometimes(
                0.5,
                iaa.Affine(translate_px={"x": (-15, 15), "y": (-15, 15)})
            ),
            iaa.Sometimes(
                0.5,
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
            ),   
            ], random_order=True)

        self.crop = RandomSizedCrop([0.80, 1.1], 224)
        self.flip = Flip(0.5)
        
        self.to_crop = to_crop


    def __call__(self, images, labels):
        if (self.to_crop):
            images, labels = self.flip(images, labels)
            for i in range(5):
                images[i], labels[i] = self.affinity(image = images[i],
                                                    segmentation_maps = labels[i][np.newaxis, :, :, np.newaxis])
                labels[i] = labels[i][0, :, :, 0]
                images, labels = self.crop(images, labels)
        else: 
            for i in range(5): 
                h, w = labels[i].shape
                images[i] = (cv2.resize(images[i], (w, h), interpolation=cv2.INTER_LINEAR))
                images[i] = images[i]/255.
                labels[i] = Image.fromarray(labels[i]).resize((w, h), resample=Image.NEAREST)
                labels[i] = np.asarray(labels[i], dtype=np.int8)
        return images, labels
        
