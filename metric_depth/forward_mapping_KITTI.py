import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import torch.nn as nn
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
to_pil_image = transforms.ToPILImage()

from depth_anything_v2.dpt import DepthAnythingV2

class DepthPrediction(nn.Module):
    def __init__(self):
        super(DepthPrediction, self).__init__()
        # 使用两个卷积层，保证输入输出维度一致
        self.conv1 = nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)  # 将输出限制在0到1之间
        return x

def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    R = torch.tensor([[0, 1], [1, 0]]).float().to(grid.device)
    XZ = torch.einsum('ij, hwj -> hwi', R, grid)  # shape = [satmap_sidelength, satmap_sidelength, 2]
    return XZ

def from_homogeneous(points, eps: float = 1e-8):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    Satmap_zoom = 18
    SatMap_original_sidelength = 512 # 0.2 m per pixel
    SatMap_process_sidelength = 512 # 0.2 m per pixel
    Default_lat = 49.015
    EPS = 1e-07
    ori_grdH, ori_grdW = 256, 1024

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    sat_img = Image.open('./sat.png')  # 替换为你的图片路径
    grd_image = cv2.imread('./grd.png')

    depth_anything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 80})
    depth_anything.load_state_dict(torch.load('/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    depth = depth_anything.infer_image(grd_image, 518)
    mask = torch.load('depth.pt').clone().detach().cpu().numpy()
    mask[mask != 0] = 1
    origin_depth = depth * mask

    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 将图片转换为 Tensor
    sat = transform(sat_img)

    # 将 BGR 转换为 RGB
    grd_image_rgb = cv2.cvtColor(grd_image, cv2.COLOR_BGR2RGB)

    # 将 NumPy 数组转换为 PIL 图像
    pil_image = Image.fromarray(grd_image_rgb)

    # 应用转换操作
    grd_image = transform(pil_image).unsqueeze(0)

    camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                            [0.0000, 482.7076, 125.0034],
                            [0.0000, 0.0000, 1.0000]]],
                        dtype=torch.float32, requires_grad=False)  # [1, 3, 3]

    camera_k[:, :1, :] = camera_k[:, :1, :] * ori_grdW / 1242  # original size input into feature get network/ output of feature get network
    camera_k[:, 1:2, :] = camera_k[:, 1:2, :] * ori_grdH / 375

    project_depth = torch.from_numpy(origin_depth).long()

    # 将 depth 转换为 one-hot 编码
    one_hot_depth = torch.nn.functional.one_hot(project_depth, num_classes=81)  # num_classes=4 表示4个类别: 0, 1, 2, 3
    # 删除第四个类别
    one_hot_depth = one_hot_depth[..., 1:].float()
    # 增加一个维度
    one_hot_depth = one_hot_depth.unsqueeze(0)

    binary_tensor = (project_depth != 0).float()  # 生成一个二值化tensor，非零为1，零为0
    # 将tensor扩展到所需形状 [3, 3, 80]
    mask = binary_tensor.unsqueeze(-1).repeat(1, 1, 80)
    # 增加一个维度
    mask = mask.unsqueeze(0)

    # predict depth
    # model = DepthPrediction()
    # depth_scores = one_hot_depth.permute(0,3,1,2)
    # output_scores = model(depth_scores).permute(0,2,3,1) * mask
    
    output_scores = one_hot_depth * mask
    weights = output_scores * torch.cumprod(torch.cat([torch.ones((output_scores.shape[0],1,output_scores.shape[2],output_scores.shape[3])), (1. - output_scores)], 1), 1)[:,:-1,:,:]
    # depth_prob = torch.softmax(depth_scores, dim=1)
    # image_polar = torch.einsum("...dhw,...hwz->...dzw", image_tensor, depth_prob)
    image_polar = torch.einsum("...dhw,...hwz->...dzw", grd_image, weights)

    f = camera_k[:, 0, 0][..., None, None]
    c = camera_k[:, 0, 2][..., None, None]

    z_max = 100
    x_max = 50
    z_min = 0
    Δ = 100 / 512

    grid_xz = make_grid(
        x_max * 2, z_max, step_y=Δ, step_x=Δ, orig_y=-50, orig_x=-x_max, y_up=False
    )
    u = from_homogeneous(grid_xz).squeeze(-1) * f + c
    # u= torch.flip(u, dims=[-1])
    z_idx = (grid_xz[..., 1] - z_min)
    z_idx = z_idx[None].expand_as(u)
    grid_polar = torch.stack([u, z_idx], -1)

    size = grid_polar.new_tensor(image_polar.shape[-2:][::-1])
    grid_uz_norm = (grid_polar * 2 / size) - 1
    # grid_uz_norm = grid_uz_norm * grid_polar.new_tensor([1, -1])  # y axis is up
    image_bev = F.grid_sample(image_polar, grid_uz_norm, align_corners=False)

    origin_image_show = to_pil_image(grd_image[0])
    origin_image_show.save('origin_image.png')
    image_polar_show = to_pil_image(image_polar[0])
    image_polar_show.save('image_polar.png')
    image_bev_show = to_pil_image(image_bev[0])
    image_bev_show.save('image_bev.png')