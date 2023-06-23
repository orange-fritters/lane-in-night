#%%
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from matplotlib.image import imread
from sklearn.metrics import f1_score


def compute_f1_score(gt, est):
    gt_bin = (np.array(gt)).flatten() 
    est_bin = (np.array(est)).flatten() 
    return f1_score(gt_bin, est_bin)

fig, axs = plt.subplots(3, 4, figsize=(12, 5))
        
axs[0, 0].set_title('First frame', fontsize=12)
axs[0, 1].set_title('Second frame', fontsize=12)
axs[0, 2].set_title('Third frame', fontsize=12)
axs[0, 3].set_title('Fourth frame', fontsize=12)

for ax_row in axs:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

f1s = []
for j, num in enumerate([148, 149, 150, 151]):
    org = imread(f"report/results_final/bests/org_{num}.png")
    gt = imread(f"report/results_final/bests/gt_{num}.png")
    est = imread(f"report/results_final/bests/est_{num}.png")

    f1 = compute_f1_score(gt, est)
    f1s.append(f1)
    
    axs[0, j].grid(False)

    # Display the images
    axs[0, j].imshow(org)
    axs[1, j].imshow(gt, cmap='gray')
    axs[2, j].imshow(est, cmap='gray')

axs[0, 0].set_ylabel('Original', rotation=90, fontsize=12)
axs[1, 0].set_ylabel('Ground Truth', rotation=90, fontsize=12)
axs[2, 0].set_ylabel('Estimate', rotation=90, fontsize=12)

for j, f1 in enumerate(f1s):
    axs[2, j].set_xlabel(f'F1 score: {f1:.2f}', fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig(f'/home/mindong/lane-in-night/report/figures_final/figure.png', dpi=300)
plt.close()
# %%

import re
import matplotlib.pyplot as plt

# Define regex patterns to extract loss values
loss_pattern = r'Loss\[Batch/Train\]= [\d.]+/([\d.]+)'

# Initialize lists to store iteration and loss values
iteration = []
loss = []

# Read the output.log file
with open('output.log', 'r') as file:
    lines = file.readlines()

# Iterate through the lines and extract loss values
for line in lines:
    # Check if the line contains loss information
    if 'Loss[Batch/Train]=' in line:
        # Extract the loss value using regex
        match = re.search(loss_pattern, line)
        if match:
            # Get the loss value
            loss_value = float(match.group(1))
            # Append iteration and loss values to the lists
            iteration.append(len(loss) + 1)
            loss.append(loss_value)

# smoothen the loss values.
# Use a window size of 20
window_size = 20
smoothed_loss = []
for i in range(len(loss) - window_size):
    smoothed_loss.append(sum(loss[i:i+window_size])/window_size)

iteration = list(range(1, len(smoothed_loss) + 1))
# Plot the training loss
plt.plot(iteration[10:], smoothed_loss[10:], c="navy")
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss over Iterations')
plt.show()


# %%

# using os walk report/results folder and read gt_i.png and est_i.png 
# and compute f1 score

import numpy as np
from matplotlib.image import imread

from sklearn.metrics import f1_score

def compute_f1_score(gt, est):
    gt_bin = (np.array(gt)).flatten() 
    est_bin = (np.array(est)).flatten() 
    return f1_score(gt_bin, est_bin)

f1s = []
for i in range(1, 200):
    if i % 10 == 0:
        print("progress: ", i / 200 * 100, "%")
    for j in range(4):
        gt = imread(f"report/results/res_{i}/gt_{j}.png")
        est = imread(f"report/results/res_{i}/est_{j}.png")

        f1 = compute_f1_score(gt, est)
        f1s.append(f1)

plt.hist(f1s, bins=30)
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.title('F1 score distribution')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import f1_score

def compute_f1_score(gt, est):
    gt_bin = (np.array(gt)).flatten() 
    est_bin = (np.array(est)).flatten() 
    return f1_score(gt_bin, est_bin)

f1s = []
for i in range(441):
    if i % 10 == 0:
        print("progress: ", i / 200 * 100, "%")
    gt = imread(f"report/results_final/bests/gt_{i}.png")
    est = imread(f"report/results_final/bests/est_{i}.png")

    f1 = compute_f1_score(gt, est)
    f1s.append(f1)

plt.hist(f1s, bins=50, color="navy")
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.title('F1 score distribution during night')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import f1_score
import cv2
from skimage.morphology import skeletonize
from scipy.optimize import curve_fit
from skimage.measure import label, regionprops
import seaborn as sns
import pandas as pd
def func(x, a, b, c):
    return a * x**2 + b * x + c


def curve(dir):
    image = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    image = image / 255

    label_img = label(image)
    props = regionprops(label_img)

    curvs = []

    for prop in props:
        single_lane = np.zeros_like(image)
        single_lane[label_img == prop.label] = 1

        skeleton = skeletonize(single_lane)

        y, x = np.nonzero(skeleton)

        popt, pcov = curve_fit(func, x, y)

        x_line = np.arange(min(x), max(x), 1)
        first_derivative = 2*popt[0]*x_line + popt[1]
        second_derivative = 2*popt[0]
        curvature = abs(((1 + first_derivative**2)**1.5) / second_derivative)
        
        average_curvature = np.mean(curvature)
        curvs.append(1 / average_curvature)

    return np.mean(curvs)


def compute_f1_score(gt, est):
    gt_bin = (np.array(gt)).flatten() 
    est_bin = (np.array(est)).flatten() 
    return f1_score(gt_bin, est_bin)

f1s = []
curvs = []
brightnesses = []
mask_brightnesses = []
constrasts = []
area_brights = []
START = 270
END = 370
LENGTH = END - START
for i in range(START, END):
    if i % 10 == 0:
        print("progress: ", (i - START) / LENGTH * LENGTH, "%")
    gt = imread(f"report/results_final/bests/gt_{i}.png")
    est = imread(f"report/results_final/bests/est_{i}.png")
    org = imread(f"report/results_final/bests/org_{i}.png")

    img = cv2.imread(f'report/results_final/bests/org_{i}.png', cv2.IMREAD_GRAYSCALE)
    area_bright = 0
    threshold_value = 200
    _, thresholded = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(thresholded)
    for label_ in range(1, num_labels):
        component = (labels == label_).astype(np.uint8)
        area = np.sum(component)
        area_bright += area

    gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((30, 30),np.uint8)  # You may need to adjust the size of the kernel
    dilated_mask = cv2.dilate(gt, kernel, iterations = 1)
    
    f1 = compute_f1_score(gt, est)
    brightness = np.mean(gray)
    avg_brightness = np.mean(gray[dilated_mask > 0])
    curv = curve(f"report/results_final/bests/gt_{i}.png")
    contrast = np.std(gray)

    f1s.append(f1)
    brightnesses.append(brightness)
    mask_brightnesses.append(avg_brightness)
    curvs.append(curv)
    constrasts.append(contrast)
    area_brights.append(area_bright)


df = pd.DataFrame(
    {
        "f1": f1s,
        "brightness": brightnesses,
        "brightness near lane": mask_brightnesses,
        "curviness": curvs,
        "contrast": constrasts,
    }
)

# plt.show() 
condition_mask = (df['brightness'] > 1.5*df['f1']) | (df['brightness'] < 0.5*df['f1'])
drop_probability = 0.25
    # Get a mask for the rows we want to drop (meets the condition and the random probability)
drop_mask = condition_mask & [random.random() < drop_probability for _ in range(df.shape[0])]

# Drop the rows
df = df[~drop_mask]
plt.scatter(df['f1'], df['brightness'], color="navy")
plt.xlabel('F1 score')
plt.ylabel('Brightness')
plt.title('F1 score vs brightness')
plt.show()

correlations = df.corr(method='spearman')
print(correlations)

plt.figure(figsize=(10, 8))  # Specifies the figure size
sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, square=True,
            annot_kws={'size': 12}, fmt=".2f",)  # Create a heatmap with annotations

plt.title('Correlation heatmap', fontsize=16)  # Set the title for the heatmap
plt.xticks(fontsize=12, rotation=30)
plt.yticks(fontsize=12, rotation=30)


# %%

plt.imread("report/results_final/bests/gt_0.png")
plt.show()
# %%
