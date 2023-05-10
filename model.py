import torch
from VGG import VGG16
from output_head import Decoder
import torch.nn as nn
import torch.nn.functional as F

class SA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        batch, height, width = mask.shape
        mask = mask.view(batch, height*width)
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask


class CVML(nn.Module):
    def __init__(self, sa_num=8, grdH=320, grdW=640, satH=512, satW=512):
        super().__init__()
        # grd
        self.vgg_grd = VGG16()
        self.grd_max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        in_dim_grd = (grdH // 32) * (grdW // 32)
        self.grd_SA = SA(in_dim_grd, num=sa_num)
        # sat
        self.sat_split = 8  # split satellite feature into 8*8 sub-volumes
        self.vgg_sat = VGG16()
        in_dim_sat = (satH // 16 // self.sat_split) * (satW // 16 // self.sat_split)
        self.sat_SA = SA(in_dim_sat, num=sa_num)
        self.dimension = sa_num

        self.costmap_decoder = Decoder()

    def forward(self, img_sat, img_grd):
        # grd
        grd_local, _, _, _, _ = self.vgg_grd(img_grd)
        grd_local = self.grd_max_pool(grd_local)
        batch, channel, g_height, g_width = grd_local.shape
        grd_w = self.grd_SA(grd_local)
        grd_local = grd_local.view(batch, channel, g_height*g_width)
        grd_global = torch.matmul(grd_local, grd_w).view(batch, -1)  # (Batch, channel*sa_num = 512*8)
        grd_global = F.normalize(grd_global, p=2, dim=1)

        # sat
        sat_local, sat512, sat256, sat128, sat64 = self.vgg_sat(img_sat)  # sat_local [Batch, 512, 32, 32]
        _, channel, s_height, s_width = sat_local.shape

        sat_global = []
        for i in range(0, self.sat_split):
            strip_horizontal = sat_local[:, :, i*s_height//self.sat_split:(i+1)*s_height//self.sat_split, :]
            sat_global_horizontal = []
            for j in range(0, self.sat_split):
                patch = strip_horizontal[:, :, :, j*s_height//self.sat_split:(j+1)*s_height//self.sat_split]
                sat_w = self.sat_SA(patch)  # Batch 16 8
                _, channel, p_height, p_width = patch.shape
                patch = patch.reshape(batch, channel, p_height*p_width)  # Batch 512 16
                # Batch 512 8 --> Batch 1, 1, 4096
                patch_global = torch.matmul(patch, sat_w).reshape(batch, 1, 1, self.dimension*channel)
                patch_global = F.normalize(patch_global, p=2, dim=-1)

                if j == 0:
                    sat_global_horizontal = patch_global
                else:
                    sat_global_horizontal = torch.cat([sat_global_horizontal, patch_global], dim=2)
            if i == 0:
                sat_global = sat_global_horizontal
            else:
                sat_global = torch.cat([sat_global, sat_global_horizontal], dim=1)
        # get matching score and logits
        # B 4096 --> B 1 1 4096 --> B 8 8 4096
        grd_global_broadcasted = torch.broadcast_to(grd_global.reshape(batch, 1, 1, grd_global.shape[-1]),
                                                    [grd_global.shape[0], self.sat_split, self.sat_split, grd_global.shape[-1]])
        matching_score = torch.sum(torch.mul(grd_global_broadcasted, sat_global), dim=-1, keepdim=True)
        cost_map = torch.cat([matching_score, sat_global], dim=3)  # Batch 8 8 4097
        logits = self.costmap_decoder(cost_map.permute(0, 3, 1, 2), sat512, sat256, sat128, sat64, sat_local)

        return logits, matching_score.permute(0, 3, 1, 2)


if __name__ == '__main__':
    model = CVML().cuda()
    input_img_grd = torch.randn(2, 3, 320, 640)
    input_img_sat = torch.randn(2, 3, 512, 512)
    result = model(input_img_sat, input_img_grd)
    print(result[0].shape)
    print(result[1].shape)


