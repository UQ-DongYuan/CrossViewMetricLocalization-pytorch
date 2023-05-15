from model import CVML
from readdata_VIGOR import VIGOR
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

load_model_path = 'checkpoint/CVMetricLocalization.pth'  # path to the trained model
val = False
dimension = 8
train_test = 'test'
area = 'same'
GT = 'Gaussian'

# setup test data loader
test_dataset = VIGOR(area, train_test, val)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# build model
model = CVML()
model.load_state_dict(torch.load(load_model_path)["state_dict"])
model.eval()
softmax = torch.nn.Softmax(dim=1)

with torch.set_grad_enabled(False):
    for i, (sat, grd, gt) in tqdm(enumerate(test_loader)):
        gaussian_gt = gt      # 1  1  512  512

        logits, matching_score = model(sat, grd)     # matching score: 1 1 8 8
        logits_reshaped = logits.reshape(logits.shape[0], 512 * 512)
        heat_map = torch.reshape(softmax(logits_reshaped), logits.shape).numpy()  # 1 1 512 512
        print(heat_map.shape)

        plt.figure(figsize=(4, 4))
        plt.imshow(matching_score[0, 0, :, :])
        plt.axis('off')
        plt.savefig('results/multi_matching_' + str(i) + '.png', bbox_inches='tight')

        plt.figure(figsize=(6, 6))
        plt.imshow(heat_map[0, 0, :, :], norm=LogNorm(vmin=1e-9, vmax=np.max(heat_map[0, 0, :, :])), alpha=0.6,
                   cmap='Reds')

        ax = plt.gca()
        ax.set_xticks(np.arange(0, 512, 512 / 8))
        ax.set_yticks(np.arange(0, 512, 512 / 8))
        ax.grid(color='w', linestyle='-', linewidth=1)
        loc_gt = np.unravel_index(gaussian_gt[0, :, :, :].argmax(), gaussian_gt[0, :, :, :].shape)
        plt.scatter(loc_gt[2], loc_gt[1], s=200, marker='^', facecolor='g', label='GT', edgecolors='white')
        loc_pred = np.unravel_index(heat_map[0, :, :, :].argmax(), heat_map[0, :, :, :].shape)
        plt.scatter(loc_pred[2], loc_pred[1], s=200, marker='*', facecolor='gold', label='Ours', edgecolors='white')
        plt.axis('off')
        plt.savefig('results/multi_heatmap_' + str(i) + '.png', bbox_inches='tight')

        if i == 4:
            break


