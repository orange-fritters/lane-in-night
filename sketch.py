import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
 
# general libs
import math
import gc
import utils.helper as helper
import numpy as np
from PIL import Image

            
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# frames = torch.rand(1, 5, 3, 224, 224)
# masks = torch.randint(0, 2, (1, 5, 1, 224, 224))

# frames = frames.to(device)
# masks = masks.to(device)

# model = STM()
# model = model.to(device)

# with torch.no_grad():
#     k4, v4 = model("memorize", frames[:, 0, :, :, :], masks[:, 0, :, :, :])
#     n2_logit = model("segment", frames[:, 1, :, :, :], k4, v4) # segment


##############MODEL CLEAR#################
# model.cpu()
# del model
# gc.collect()
# torch.cuda.empty_cache()
##########################################

f_list = [np.random.rand(224, 224, 3) for i in range(5)]
f_list = np.array(f_list)
f_list.shape

np.transpose(f_list.copy(), (0, 3, 1, 2)).shape # (5, 3, 224, 224)