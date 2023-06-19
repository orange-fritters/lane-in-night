import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import f1_score
from PIL import Image

def compute_f1_score(gt, est):
    gt_bin = (np.array(gt)).flatten() 
    est_bin = (np.array(est)).flatten() 
    return f1_score(gt_bin, est_bin)


def plot_images():
    save_dir = '/home/mindong/lane-in-night/report/results_night_fine_tuned'
    for i in range(50):
        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}th image...")
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
        for j in range(4):
            org = imread(f"{save_dir}/res_{i}/org_{j}.png")
            gt = imread(f"{save_dir}/res_{i}/gt_{j}.png")
            est = imread(f"{save_dir}/res_{i}/est_{j}.png")

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
        plt.savefig(f'/home/mindong/lane-in-night/report/figures_night_fine_tuned/figure_{i}.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    plot_images()
