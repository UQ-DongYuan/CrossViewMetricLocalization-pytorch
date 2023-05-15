from model import CVML
from readdata_VIGOR import VIGOR
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm

load_model_path = 'checkpoint/CVMetricLocalization.pth'  # path to the trained model
val = False
dimension = 8
train_test = 'test'
area = 'same'

# setup test data loader
test_dataset = VIGOR(area, train_test, val)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# build model
model = CVML().cuda()
print("load model ...")
model.load_state_dict(torch.load(load_model_path)["state_dict"])
model.eval()
softmax = torch.nn.Softmax(dim=1)

with torch.set_grad_enabled(False):
    distance = []
    probability_at_gt = []
    for i, (sat, grd, gt) in tqdm(enumerate(test_loader)):
        sat = sat.cuda()
        grd = grd.cuda()
        gaussian_gt = gt.cuda()  # 1  1  512  512

        logits, matching_score = model(sat, grd)  # logits: B 1 512 512
        logits_reshaped = logits.reshape(logits.shape[0], 512 * 512)
        heat_map = torch.reshape(softmax(logits_reshaped), logits.shape)

        for batch_idx in range(gaussian_gt.shape[0]):
            current_gt = gaussian_gt[batch_idx, :, :, :].cpu().detach().numpy()  # B 1 512 512
            loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
            current_pred = heat_map[batch_idx, :, :, :].cpu().detach().numpy()
            loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
            distance.append(np.sqrt((loc_gt[1] - loc_pred[1]) ** 2 + (loc_gt[2] - loc_pred[2]) ** 2))
            probability_at_gt.append(current_pred[loc_gt[1], loc_gt[2]])

    print('mean distance error',
          np.mean(distance) * 0.1425)  # 0.1425 is the ground distance per pixel. 0.1425 = 0.114/512*640
    print('median distance error', np.median(distance) * 0.1425)
    print('mean distance error', np.mean(probability_at_gt))
    print('mean distance error', np.median(probability_at_gt))

