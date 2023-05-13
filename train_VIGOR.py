from model import CVML
from readdata_VIGOR import VIGOR
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

area = 'same'
learning_rate = 1e-5
start_epoch = 0
end_epoch = 50
batch_size = 8
keep_prob_val = 0.8
dimension = 8
beta = 1e4
temperature = 0.1
label = 'VIGOR_'+area
save_model_path = './models/'
train_data_size = 42087   # training iterations: 42087 // 8 = 5260
val_data_size = 10522     # val  iterations: 10522 // 8 = 1315
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class ContrastiveLoss(nn.Module):
    def __init__(self, tem=temperature):
        super().__init__()
        self.temperature = tem

    def forward(self, scores, labels):
        exp_scores = torch.exp(scores / self.temperature)
        bool_mask = labels.ge(1e-2)
        denominator = torch.sum(exp_scores, [1, 2, 3], keepdim=True)

        inner_element = torch.log(torch.masked_select(exp_scores/denominator, bool_mask))

        return -torch.sum(inner_element*torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

class SoftmaxCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, logits, dim=-1):
        return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)

def main():
    # setup train/val dataset
    train_dataset = VIGOR(area=area, train_test='train', val=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = VIGOR(area=area, train_test='train', val=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # build model
    model = CVML().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    bottleneck_loss = ContrastiveLoss()
    heatmap_loss = SoftmaxCrossEntropyWithLogits()
    best_distance_error = 9999

    for epoch_idx in range(start_epoch, end_epoch):
        model.train()

        epoch_loss = []
        iteration = 0

        for iteration, (sat, grd, gt) in tqdm(enumerate(train_dataloader)):
            sat_img = sat.to(device)
            grd_img = grd.to(device)
            gt = gt.to(device)  # B 1 512 512
            gt_bottleneck = torch.max_pool2d(gt, kernel_size=64, stride=64)  # B 1 8 8

            optimizer.zero_grad()
            logits, matching_score = model(sat_img, grd_img)  # matching_score: B 1 8 8

            logits_reshaped = logits.reshape(logits.shape[0], 512*512)
            gt_reshape = gt.reshape(logits.shape[0], 512*512)
            gt_reshape = gt_reshape / torch.sum(gt_reshape, dim=1, keepdim=True)
            loss_heatmap = torch.mean(heatmap_loss(gt_reshape, logits_reshaped))
            loss_bottleneck = bottleneck_loss(matching_score, gt_bottleneck)

            loss = loss_heatmap + loss_bottleneck * beta
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        curr_training_loss = sum(epoch_loss) / (iteration + 1)
        train_file = 'training_loss.txt'
        with open(train_file, 'a') as file:
            file.write(f"Epoch {epoch_idx} Training Loss: {curr_training_loss}" + '\n')

        print(f"Epoch {epoch_idx} Training Loss: {curr_training_loss}")

        print('validate....')
        model.eval()
        val_epoch_loss = []
        distance = []
        softmax = torch.nn.Softmax(dim=1)

        with torch.set_grad_enabled(False):
            for i, (val_sat, val_grd, val_gt) in tqdm(enumerate(val_dataloader)):
                val_sat = val_sat.to(device)
                val_grd = val_grd.to(device)
                val_gt = val_gt.to(device)
                val_gt_bottleneck = torch.max_pool2d(val_gt, kernel_size=64, stride=64)

                val_logits, val_matching_score = model(val_sat, val_grd)

                val_logits_reshaped = val_logits.reshape(val_logits.shape[0], 512 * 512)
                val_gt_reshape = val_gt.reshape(val_logits.shape[0], 512 * 512)
                val_gt_reshape = val_gt_reshape / torch.sum(val_gt_reshape, dim=1, keepdim=True)
                val_loss_heatmap = torch.mean(heatmap_loss(val_gt_reshape, val_logits_reshaped))
                val_loss_bottleneck = bottleneck_loss(val_matching_score, val_gt_bottleneck)

                val_loss = val_loss_heatmap + val_loss_bottleneck * beta
                val_epoch_loss.append(val_loss.item())

                val_heatmap = softmax(val_logits_reshaped).reshape(val_logits.shape)
                for batch_idx in range(batch_size):
                    current_gt = val_gt[batch_idx, :, :, :].cpu().detach().numpy()   # B 1 512 512
                    loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                    current_pred = val_heatmap[batch_idx, :, :, :].cpu().detach().numpy()
                    loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
                    distance.append(np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2))

            curr_val_loss = sum(val_epoch_loss) / (i+1)
            distance_error = np.mean(distance)
            val_loss_file = 'val_loss.txt'
            val_distance_error = 'val_distance_error'
            with open(val_loss_file, 'a') as file:
                file.write(f"Epoch {epoch_idx} val Loss: {curr_val_loss}" + '\n')
            with open(val_distance_error, 'a') as file:
                file.write(f"Epoch {epoch_idx} val distance error: {distance_error}" + '\n')
            print(f"Epoch {epoch_idx} mean distance error on validation set: {distance_error}")
            print(f"Epoch {epoch_idx} validation Loss: {curr_val_loss}")
        if distance_error < best_distance_error:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            model_path = "checkpoint/CVMetricLocalization.pth"
            save_checkpoint(
                {"state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()
                 },
                model_path
            )
            best_distance_error = distance_error
            print(f"Model saved at distance error: {best_distance_error}")


if __name__ == '__main__':
    main()

