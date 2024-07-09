import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from depth_anything_v2.dpt import DepthAnythingV2

Satmap_zoom = 18
SatMap_original_sidelength = 512 # 0.2 m per pixel
SatMap_process_sidelength = 512 # 0.2 m per pixel
Default_lat = 49.015
EPS = 1e-07
ori_grdH, ori_grdW = 256, 1024

def grid_sample(image, optical, jac=None):
    # values in optical within range of [0, H], and [0, W]
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(N, 1, H, W)
    iy = optical[..., 1].view(N, 1, H, W)

    with torch.no_grad():
        ix_nw = torch.floor(ix)  # north-west  upper-left-x
        iy_nw = torch.floor(iy)  # north-west  upper-left-y
        ix_ne = ix_nw + 1        # north-east  upper-right-x
        iy_ne = iy_nw            # north-east  upper-right-y
        ix_sw = ix_nw            # south-west  lower-left-x
        iy_sw = iy_nw + 1        # south-west  lower-left-y
        ix_se = ix_nw + 1        # south-east  lower-right-x
        iy_se = iy_nw + 1        # south-east  lower-right-y

        torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

        torch.clamp(ix_se, 0, IW -1, out=ix_se)
        torch.clamp(iy_se, 0, IH -1, out=iy_se)

    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0

    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:

        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # assert jac.shape[1:] == [N, H, W, 2]
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        if torch.any(torch.isnan(jac)) or torch.any(torch.isnan(dout_dpxy)):
            print('Nan occurs')

        return out_val, jac_new1 #jac_new1 #jac_new.permute(4, 0, 1, 2, 3)
    else:
        return out_val, None

def grd_img2cam(grd_H, grd_W, ori_grdH, ori_grdW, depth):
        
    ori_camera_k = torch.tensor([[[582.9802,   0.0000, 496.2420],
                                    [0.0000, 482.7076, 125.0034],
                                    [0.0000,   0.0000,   1.0000]]], 
                                dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
    
    camera_height = 1.65

    camera_k = ori_camera_k.clone()
    camera_k[:, :1, :] = ori_camera_k[:, :1,
                            :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
    camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
    camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

    v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                            torch.arange(0, grd_W, dtype=torch.float32))
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0)  # [1, grd_H, grd_W, 3]
    xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

    w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > EPS, xyz_w[..., 1:2],
                                    EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
    xyz_grd = xyz_w * depth  # [1, grd_H, grd_W, 3] under the grd camera coordinates
    # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)

    mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]


    # test
    # 提取 x 和 z 坐标
    test = xyz_grd
    x_coords = test[0, :, :, 0]  # 形状 [256, 1024]
    y_coords = test[0, :, :, 1]
    z_coords = test[0, :, :, 2]  # 形状 [256, 1024]
    # 创建 tensorB，形状为 [1, 256, 256, 2]
    tensorB = torch.zeros(512, 512, 1).detach()

    # 填充 tensorB，使用 tensorA 中的 x 和 z 坐标作为索引
    for i in range(256):
        for j in range(1024):
            x_idx = int(x_coords[i, j] / get_meter_per_pixel() + 512 / 2)
            z_idx = int(z_coords[i, j] / get_meter_per_pixel() + 512 / 2)
            if(x_idx < 512 and x_idx > 0 and z_idx < 512 and z_idx > 0 and y_coords[i, j] > tensorB[x_idx, z_idx, 0]):
                tensorB[x_idx, z_idx, 0] = y_coords[i, j]

    return xyz_grd, mask, xyz_w

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel


def grd2cam2world2sat(xyz_grds, satmap_sidelength=512, require_jac=False, gt_depth=None, level=0):
    '''
    realword: X: south, Y:down, Z: east
    camera: u:south, v: down from center (when heading east, need to rotate heading angle)
    Args:
        ori_shift_u: [B, 1]
        ori_shift_v: [B, 1]
        heading: [B, 1]
        XYZ_1: [H,W,4]
        ori_camera_k: [B,3,3]
        grd_H:
        grd_W:
        ori_grdH:
        ori_grdW:

    Returns:
    '''
    B = 1
    heading = torch.ones(1, 1).to('cuda') * (-0.8362) * 10 / 180 * np.pi
    shift_u = torch.ones(1, 1).to('cuda') * (0.2122) * 20
    shift_v = torch.ones(1, 1).to('cuda') * (-0.7794) * 20

    # heading = torch.zeros(1, 1).to('cuda') * (-0.8362) * 10 / 180 * np.pi
    # shift_u = torch.zeros(1, 1).to('cuda') * (0.2122) * 20
    # shift_v = torch.zeros(1, 1).to('cuda') * (-0.7794) * 20

    cos = torch.cos(heading)
    sin = torch.sin(heading)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, 9]
    R = R.view(B, 3, 3)  # shape = [B, N, 3, 3]
    # this R is the inverse of the R in G2SP

    camera_height = 1.65 # meters
    # camera offset, shift[0]:east,Z, shift[1]:north,X
    height = camera_height * torch.ones_like(shift_u[:, :1])
    T0 = torch.cat([shift_v, height, -shift_u], dim=-1)  # shape = [B, 3]
    # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
    # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
    T = torch.sum(-R * T0[:, None, :], dim=-1)   # [B, 3]

    # The above R, T define transformation from camera to world

    xyz_grd = xyz_grds[level][0].detach().to('cuda').repeat(B, 1, 1, 1)
    mask = xyz_grds[level][1].detach().to('cuda').repeat(B, 1, 1)  # [B, grd_H, grd_W]
    grd_H, grd_W = xyz_grd.shape[1:3]

    xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
    # [B, grd_H, grd_W, 3]
    # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
    R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device='cuda', requires_grad=True)\
        .reshape(2, 3)
    zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
    # [B, grd_H, grd_W, 2]
    # assert zx == zx0

    meter_per_pixel = get_meter_per_pixel()
    meter_per_pixel *= 512 / satmap_sidelength
    sat_uv = zx/meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv

    return sat_uv, mask, None, None, None

def showDepth(depth, raw_image):
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    output_path = './depth.png'

    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([raw_image, split_region, depth])
    
    cv2.imwrite(output_path, combined_result)

def project_map_to_grd(sat, depth):
    xyz_grds = []
    depth = depth * 100 / 80
    xyz_grd, mask, xyz_w = grd_img2cam(ori_grdH, ori_grdW, ori_grdH, ori_grdW, depth)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
    xyz_grds.append((xyz_grd, mask, xyz_w))

    # for level in range(4):
    #     grd_H, grd_W = ori_grdH/(2**(3-level)), ori_grdW/(2**(3-level))

    #     xyz_grd, mask, xyz_w = grd_img2cam(grd_H, grd_W, ori_grdH,
    #                                         ori_grdW)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
    #     xyz_grds.append((xyz_grd, mask, xyz_w))

    uv, mask, jac_shiftu, jac_shiftv, jac_heading = grd2cam2world2sat(xyz_grds)

    sat_project, new_jac = grid_sample(sat, uv, None)
    sat_project = sat_project * mask[:, None, :, :]
    sat_img = transforms.ToPILImage()(sat_project[0])
    sat_img.save('sat_proj.png')
    return sat_project


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str, default='/home/wangqw/video_dataset/KITTI_street')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/home/wangqw/video_dataset/KITTI_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=80)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 读取图片
    sat_img = Image.open('./sat.png')  # 替换为你的图片路径

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    grd_image = cv2.imread('./grd.png')
    depth = depth_anything.infer_image(grd_image, args.input_size)
    mask = torch.load('depth.pt').clone().detach().cpu().numpy()
    mask[mask != 0] = 1
    depth = depth * mask
    showDepth(depth, grd_image)
    depth = torch.tensor(depth, dtype=torch.float32)
    depth = depth.unsqueeze(0).unsqueeze(-1)

    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 将图片转换为 Tensor
    sat = transform(sat_img).to('cuda')

    sat_project = project_map_to_grd(sat.unsqueeze(0), depth)

