import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import torch.nn as nn
from typing import Optional

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage()

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
    xyz_grd = xyz_w * w
    # xyz_grd = xyz_w * depth  # [1, grd_H, grd_W, 3] under the grd camera coordinates
    # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)

    mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

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


def sat2world(grd_H, grd_W, depth, img):
    ori_camera_k = torch.tensor([[[582.9802,   0.0000, 496.2420],
                                      [0.0000, 482.7076, 125.0034],
                                      [0.0000,   0.0000,   1.0000]]], 
                                    dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
    meter_per_pixel = get_meter_per_pixel()
    camera_k = ori_camera_k.clone()
    camera_k[:, :1, :] = ori_camera_k[:, :1,
                            :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
    camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
    camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

    v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                            torch.arange(0, grd_W, dtype=torch.float32))
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0)  # [1, grd_H, grd_W, 3]
    xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]
    
    xyz_grd = xyz_w * torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1)
    xyz_grd = xyz_grd / meter_per_pixel # [1, 256, 1024, 3]
    xyz_grd[:,:,:,0:1] += 50
    xyz_grd[:,:,:,2:-1] += 50
    B, H, W, C = xyz_grd.shape
    xyz_geo = xyz_grd.view(B*H*W, -1)
    img_feat = img.view(B*H*W, -1)
    kept = (xyz_geo[:,0] >= 0) & (xyz_geo[:,0] <= 100) & (xyz_geo[:,2] >= 0) & (xyz_geo[:,2] <= 100)

    xyz_geo_kept = xyz_geo[kept]
    xyz_geo_kept = torch.floor(xyz_geo_kept)
    img_kept = img_feat[kept]
    rank = xyz_geo_kept[:, 0] * 100 + xyz_geo_kept[:, 2] * 100
    sorts = rank.argsort()

    xyz_geo_kept = xyz_geo_kept[sorts]
    img_kept = img_kept[sorts]
    rank = rank[sorts]

    x = img_kept.cumsum(0)
    kept = torch.ones(x.shape[0], dtype=torch.bool)
    kept[:-1] = (rank[1:] != rank[:-1])

    x, xyz_geo_kept = x[kept], xyz_geo_kept[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    final = torch.zeros(1,3,100,100)
    final[0, :, xyz_geo_kept[:, 0].long(), xyz_geo_kept[:, 2].long()] = x.to('cpu').t()
    final_normalized = F.normalize(final, p=2, dim=1)
    final_img = transforms.functional.to_pil_image(final_normalized.squeeze(0), mode='RGB')
    final_img.save('final_img.png')
    return xyz_grd

def World2GrdImgPixCoordinates(ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W,
                                   ori_grdH, ori_grdW):
    # realword: X: south, Y:down, Z: east
    # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
    # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
    B = ori_heading.shape[0]
    shift_u_meters = ori_shift_u
    shift_v_meters = ori_shift_v
    heading = ori_heading

    cos = torch.cos(-heading)
    sin = torch.sin(-heading)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
    R = R.view(B, 3, 3)  # shape = [B,3,3]

    camera_height = 1.65
    # camera offset, shift[0]:east,Z, shift[1]:north,X
    height = camera_height * torch.ones_like(shift_u_meters)
    T0 = torch.cat([shift_u_meters, height, -shift_v_meters], dim=-1)  # shape = [B, 3]
    # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
    # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
    T = torch.sum(-R * T0[:, None, :], dim=-1)   # [B, 3]

    # P = K[R|T]
    camera_k = ori_camera_k.clone()
    camera_k[:, :1, :] = ori_camera_k[:, :1,
                            :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
    camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
    P = camera_k @ torch.cat([R, T], dim=-1)

    uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
    # only need view in front of camera ,Epsilon = 1e-6
    uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
    uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

    mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)

    return uv, mask

def showDepth(depth, raw_image):
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    output_path = './depth.png'

    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([raw_image, split_region, depth])
    
    cv2.imwrite(output_path, combined_result)

def project_grd_to_map(grd_f, depth):
    '''
    grd_f: [B, C, H, W]
    grd_c: [B, 1, H, W]
    shift_u: [B, 1]
    shift_v: [B, 1]
    heading: [B, 1]
    camera_k: [B, 3, 3]
    satmap_sidelength: scalar
    ori_grdH: scalar
    ori_grdW: scalar
    '''

    B, C, H, W = grd_f.size()
    
    heading = torch.ones(1, 1).to('cuda') * (-0.8362) * 10 / 180 * np.pi
    shift_u = torch.ones(1, 1).to('cuda') * (0.2122) * 20
    shift_v = torch.ones(1, 1).to('cuda') * (-0.7794) * 20

    camera_k = torch.tensor([[[582.9802,   0.0000, 496.2420],
                                [0.0000, 482.7076, 125.0034],
                                [0.0000,   0.0000,   1.0000]]], 
                            dtype=torch.float32, requires_grad=True).to('cuda')  # [1, 3, 3]
    
    XYZ_1 = sat2world(256, 1024, depth, grd_f)  # [ sidelength,sidelength,4]
    uv, mask = World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k,
                                                H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
    # [B, H, W, 2], [2, B, H, W, 2], [1, B, H, W, 2]

    grd_f_trans, _ = grid_sample(grd_f, uv, jac=None)
    grd_img = to_pil_image(grd_f_trans[0])
    grd_img.save('porject_grd.png')
    return grd_f_trans


def predict_sat_height(image_tensor, camera_k, depth, sat_width=512):
    torch.manual_seed(42)
    origin_image_tensor = image_tensor.clone()
    B, C, grd_H, grd_W = image_tensor.shape
    meter_per_pixel = get_meter_per_pixel()
    camera_k = camera_k.clone()
    camera_k[:, :1, :] = camera_k[:, :1,
                            :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
    camera_k[:, 1:2, :] = camera_k[:, 1:2, :] * grd_H / ori_grdH
    # meter_per_pixel = 1
    image_tensor = image_tensor.permute(0,2,3,1)
    # test
    # grd_image_width = 3
    # grd_image_height = 3
    # sat_width = 5 # pixel
    # Camera_height = 10 #meter
    # image_tensor = torch.randint(0, 256, (grd_image_height, grd_image_width, 3), dtype=torch.uint8).unsqueeze(0)
    # camera_k = torch.tensor([[[1, 0, 1], [0, 1, 1], [0, 0, 1]]], dtype=torch.float32)
    # depth = torch.tensor([
    #     [1,1,1],
    #     [1,2,1],
    #     [1,2,1]
    # ])
    # depth = torch.tensor([
    #     [1,2,1,1,2],
    #     [1,3,1,2,1],
    #     [2,1,2,4,1],
    #     [1,3,1,2,1]
    # ])

    camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

    v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                            torch.arange(0, grd_W, dtype=torch.float32))
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to('cuda')
    xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]


    depth = depth.unsqueeze(0).unsqueeze(-1)
    depth = F.interpolate(depth.permute(0,3,1,2), size=(grd_H, grd_W), mode='bilinear', align_corners=False).permute(0,2,3,1)
    # xyz_grd = xyz_w * depth / meter_per_pixel
    xyz_grd = xyz_w * depth * 1.2

    # xyz_grd = xyz_grd.long()
    # xyz_grd[:,:,:,0:1] += sat_width // 2
    # xyz_grd[:,:,:,2:3] += sat_width // 2
    B, H, W, C = xyz_grd.shape
    xyz_grd = xyz_grd.view(B*H*W, -1)
    xyz_grd[:, 0] = xyz_grd[:, 0] / meter_per_pixel
    xyz_grd[:, 2] = xyz_grd[:, 2] / meter_per_pixel
    xyz_grd[:, 0] = xyz_grd[:, 0].long()
    xyz_grd[:, 2] = xyz_grd[:, 2].long()

    kept = (xyz_grd[:,0] >= -(sat_width // 2)) & (xyz_grd[:,0] <= (sat_width // 2) - 1) & (xyz_grd[:,2] >= -(sat_width // 2)) & (xyz_grd[:,2] <= (sat_width // 2) - 1)

    xyz_grd_kept = xyz_grd[kept]
    image_tensor_kept = image_tensor.view(B*H*W, -1)[kept]

    max_height = xyz_grd_kept[:,1].max()

    xyz_grd_kept[:,0] = xyz_grd_kept[:,0] + sat_width // 2
    xyz_grd_kept[:,1] = max_height - xyz_grd_kept[:,1]
    xyz_grd_kept[:,2] = xyz_grd_kept[:,2] + sat_width // 2
    xyz_grd_kept = xyz_grd_kept[:,[2,0,1]]
    rank = torch.stack((xyz_grd_kept[:, 0] * sat_width + xyz_grd_kept[:, 1] + 1, xyz_grd_kept[:, 2]), dim=1)
    sorts_second = torch.argsort(rank[:, 1])
    xyz_grd_kept = xyz_grd_kept[sorts_second]
    image_tensor_kept = image_tensor_kept[sorts_second]
    sorted_rank = rank[sorts_second]
    sorts_first = torch.argsort(sorted_rank[:, 0], stable=True)
    xyz_grd_kept = xyz_grd_kept[sorts_first]
    image_tensor_kept = image_tensor_kept[sorts_first]
    sorted_rank = sorted_rank[sorts_first]
    kept = torch.ones_like(sorted_rank[:, 0])
    kept[:-1] = sorted_rank[:, 0][:-1] != sorted_rank[:, 0][1:]
    res_xyz = xyz_grd_kept[kept.bool()]
    res_image = image_tensor_kept[kept.bool()]
    
    # grd_image_index = torch.cat((-res_xyz[:,1:2] + grd_image_width - 1,-res_xyz[:,0:1] + grd_image_height - 1), dim=-1)
    final = torch.zeros(1,sat_width,sat_width,3).to(torch.float32).to('cuda')
    sat_height = torch.zeros(1,sat_width,sat_width,1).to(torch.float32).to('cuda')
    final[0,res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_image.to(torch.float32)

    res_xyz[:,2][res_xyz[:,2] < 1e-1] = 1e-1
    sat_height[0,res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_xyz[:,2].unsqueeze(-1)
    sat_height = sat_height.permute(0,3,1,2)

    # torch.save(sat_height, 'sat_height.pt')

    # visulize
    # height_map = sat_height[0].squeeze(0)  # 现在形状为 [256, 1024]
    # plt.imshow(height_map.cpu().detach().numpy(), cmap='viridis')  # 使用 'viridis' 映射显示颜色
    # plt.colorbar(label='Satellite Height')
    # plt.title('Height Map Visualization')
    # plt.savefig('pred_height_img.png')
    # plt.close()  
    # # 去掉 batch 维度，形状变为 [3, 3, 3]
    # project_grd_img = to_pil_image(final[0].permute(2,0,1))
    # project_grd_img.save('project_grd_img.png')

    # project_grd_img = to_pil_image(origin_image_tensor.squeeze(0))
    # project_grd_img.save('grd_image_tensor.png')

    return sat_height, final.permute(0,3,1,2)



    # 生成3D点云
    # colors = image_tensor.view(B*H*W, -1)
    # # 创建颜色字符串列表
    # color_strings = ['rgb({}, {}, {})'.format(r, g, b) for r, g, b in colors]

    # # 使用 plotly 绘制点云
    # fig = go.Figure(data=[go.Scatter3d(
    #     x=xyz_grd[:, 0],
    #     y=xyz_grd[:, 1],
    #     z=xyz_grd[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=3,
    #         color=color_strings,  # 设置颜色
    #     )
    # )])

    # # 设置坐标轴标签和刻度间隔
    # fig.update_layout(scene=dict(
    #     xaxis=dict(title='X', tick0=0, dtick=1),
    #     yaxis=dict(title='Y', tick0=0, dtick=1),
    #     zaxis=dict(title='Z', tick0=0, dtick=1)
    # ))

    # # 显示图像
    # fig.show()

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
    return grid

def from_homogeneous(points, eps: float = 1e-8):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)

raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)

class DepthPrediction(nn.Module):
    def __init__(self):
        super(DepthPrediction, self).__init__()
        # 定义一个简单的卷积层
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)
        
        # 定义输出层，将通道数从 3 转换为 1，用于输出深度
        self.conv2 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        identity = x  # 跳跃连接的输入
        
        # 卷积 -> 批归一化 -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 跳跃连接：将输入 identity 加到输出
        out += identity
        
        # 通过输出层，将通道数变为 1
        out = self.conv2(out)
        out = F.softmax(out, dim=1)
        return out


def test():
    torch.manual_seed(42)
    # origin_image_tensor = image_tensor.clone()
    # B, C, grd_H, grd_W = image_tensor.shape
    # meter_per_pixel = get_meter_per_pixel()
    # camera_k = camera_k.clone()
    # camera_k[:, :1, :] = camera_k[:, :1,
    #                         :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
    # camera_k[:, 1:2, :] = camera_k[:, 1:2, :] * grd_H / ori_grdH
    # # meter_per_pixel = 1
    # image_tensor = image_tensor.permute(0,2,3,1)
    # test
    grd_image_width = 3
    grd_image_height = 3
    sat_width = 5 # pixel
    Camera_height = 10 #meter
    image_tensor = torch.tensor(
        [[[
          [1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [1.0000, 1.0000, 1.0000]],

         [[1.0000, 0.0000, 1.0000],
          [1.0000, 0.0000, 1.0000],
          [0.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]]]], dtype=torch.float32)
    
    depth = torch.tensor([
        [2,0,2],
        [2,0,2],
        [2,1,2]
    ])
    # one-hot
    # # 将 depth 转换为 one-hot 编码
    # one_hot_result = torch.nn.functional.one_hot(depth, num_classes=4)  # num_classes=4 表示4个类别: 0, 1, 2, 3
    # # 删除第四个类别
    # one_hot_result = one_hot_result[..., 1:].float()
    # # 增加一个维度
    # pixel_scales = one_hot_result.unsqueeze(0)

    # probability
    # 理想的深度值列表
    ideal_depth_values = torch.arange(3).type(torch.float32) + 1
    # 计算每个元素到理想深度值的差异，得到误差矩阵
    # 使用 torch.abs(depth.unsqueeze(-1) - ideal_depth_values)
    error_matrix = torch.abs(depth.unsqueeze(-1) - ideal_depth_values)
    # # 计算负的平方误差，并应用softmax以转换为概率分布
    # probability_tensor = torch.exp(-error_matrix**2)
    # probability_tensor = probability_tensor / probability_tensor.sum(dim=-1, keepdim=True)
    # # 为了处理可能出现的深度值为0的情况
    # # 如果 depth == 0, 则设置对应的one-hot向量为 [0, 0, 0]
    # pixel_scales = torch.where(depth.unsqueeze(-1) == 0, torch.zeros_like(probability_tensor), probability_tensor)
    # 增加一个维度
    pixel_scales = F.softmax(-error_matrix**2, dim=-1)
    pixel_scales = pixel_scales.unsqueeze(0)

    camera_k = torch.tensor([[[1, 0.0000, 1.5],
                              [0.0000, 1, 1.5],
                              [0.0000, 0.0000, 1.5]]],
                    dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
    # z_steps = torch.arange(1, 6, 1)
    # scale_steps = camera_k[:,1,None,1] * 10 / z_steps.flip(-1)
    # log_scale_steps = torch.log2(scale_steps)
    # scale_min = 0
    # scale_max = 4
    # log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
    # log_scale_norm = log_scale_norm * 2 - 1
    # indices = log_scale_norm.unsqueeze(-1)
    # indices = torch.stack([torch.zeros_like(indices), indices], -1)
    # pixel_scales = torch.tensor([[[0,0,1],[0,0,0],[0,0,1]],[[0,1,0],[0,0,0],[0,1,0]],[[1,0,0],[1,0,0],[1,0,0]]], dtype=torch.float32).unsqueeze(0)
    # mask = torch.tensor([[[1,1,1],[0,0,0],[1,1,1]],[[1,1,1],[0,0,0],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]], dtype=torch.float32).unsqueeze(0)
    # 将非0元素变为1，0元素保持为0
    binary_tensor = (depth != 0).float()  # 生成一个二值化tensor，非零为1，零为0

    # 将tensor扩展到所需形状 [3, 3, 3]
    mask = binary_tensor.unsqueeze(-1).repeat(1, 1, 3)

    # 增加一个维度
    mask = mask.unsqueeze(0)
    
    # residual_scales = torch.cat((torch.zeros_like(pixel_scales[:,:1,:,:]), torch.cumsum(pixel_scales, dim=1)[:,:-1,:,:]), dim=1)
    # res_scales = pixel_scales - residual_scales
    # values = res_scales.flatten(1, 2).unsqueeze(-1)
    # values = pixel_scales.flatten(1, 2).unsqueeze(-1)
    # depth_scores = F.grid_sample(values, indices, align_corners=True)
    # depth_scores = depth_scores.reshape(
    #         pixel_scales.shape[:-1] + (len(z_steps),)
    # )
    model = DepthPrediction()
    depth_scores = pixel_scales.permute(0,3,1,2)
    output_scores = model(depth_scores).permute(0,2,3,1) * mask
    # output_scores = pixel_scales * mask
    # dists = torch.ones_like(depth_scores)
    # alpha = raw2alpha(depth_scores, dists)
    ideal_depth_values = ideal_depth_values.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 3, 1)
    weights = output_scores * torch.cumprod(torch.cat([torch.ones((output_scores.shape[0],1,output_scores.shape[2],output_scores.shape[3])), (1. - output_scores)], 1), 1)[:,:-1,:,:]
    # depth_prob = torch.softmax(depth_scores, dim=1)
    # image_polar = torch.einsum("...dhw,...hwz->...dzw", image_tensor, depth_prob)
    image_polar = torch.einsum("...dhw,...hwz->...dzw", image_tensor, weights)

    f = camera_k[:, 0, 0][..., None, None]
    c = camera_k[:, 0, 2][..., None, None]

    z_max = 6
    x_max = 3
    z_min = 0
    Δ = 1

    grid_xz = make_grid(
        x_max * 2 + Δ, z_max, step_y=Δ, step_x=Δ, orig_y=0, orig_x=-x_max, y_up=False
    )
    u = from_homogeneous(grid_xz).squeeze(-1) * f + c
    z_idx = (grid_xz[..., 1] - z_min) / Δ
    z_idx = z_idx[None].expand_as(u)
    grid_polar = torch.stack([u, z_idx], -1)

    size = grid_polar.new_tensor(image_polar.shape[-2:][::-1])
    grid_uz_norm = (grid_polar * 2 / size) - 1
    # grid_uz_norm = grid_uz_norm * grid_polar.new_tensor([1, -1])  # y axis is up
    image_bev = F.grid_sample(image_polar, grid_uz_norm, align_corners=False)

    origin_image_show = to_pil_image(image_tensor[0])
    origin_image_show.save('origin_image.png')
    image_polar_show = to_pil_image(image_polar[0])
    image_polar_show.save('image_polar.png')
    image_bev_show = to_pil_image(image_bev[0])
    image_bev_show.save('image_bev.png')
    # visulize
    # height_map = sat_height[0].squeeze(0)  # 现在形状为 [256, 1024]
    # plt.imshow(height_map.cpu().detach().numpy(), cmap='viridis')  # 使用 'viridis' 映射显示颜色
    # plt.colorbar(label='Satellite Height')
    # plt.title('Height Map Visualization')
    # plt.savefig('pred_height_img.png')
    # plt.close()  
    # # 去掉 batch 维度，形状变为 [3, 3, 3]
    # project_grd_img = to_pil_image(final[0].permute(2,0,1))
    # project_grd_img.save('project_grd_img.png')

    # project_grd_img = to_pil_image(origin_image_tensor.squeeze(0))
    # project_grd_img.save('grd_image_tensor.png')

if __name__ == '__main__':
    test()
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
    grd_image = cv2.imread('./grd.png')

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    depth = depth_anything.infer_image(grd_image, args.input_size)
    mask = torch.load('depth.pt').clone().detach().cpu().numpy()
    mask[mask != 0] = 1
    depth = depth * mask
    # showDepth(depth, grd_image)
    # depth = torch.tensor(depth, dtype=torch.float32)
    # depth = depth.unsqueeze(0).unsqueeze(-1)

    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 将图片转换为 Tensor
    sat = transform(sat_img).to('cuda')

    # 将 BGR 转换为 RGB
    grd_image_rgb = cv2.cvtColor(grd_image, cv2.COLOR_BGR2RGB)

    # 将 NumPy 数组转换为 PIL 图像
    pil_image = Image.fromarray(grd_image_rgb)

    # 应用转换操作
    grd_image = transform(pil_image).to('cuda')

    ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                            [0.0000, 482.7076, 125.0034],
                            [0.0000, 0.0000, 1.0000]]],
                        dtype=torch.float32, requires_grad=False, device='cuda')  # [1, 3, 3]

    ori_camera_k[:, :1, :] = ori_camera_k[:, :1,
                            :] * ori_grdW / 1242  # original size input into feature get network/ output of feature get network
    ori_camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * ori_grdH / 375

    predict_sat_height(grd_image.unsqueeze(0).to('cuda'), ori_camera_k, torch.from_numpy(depth).to('cuda'))
    sat_project = project_grd_to_map(grd_image.unsqueeze(0), depth.to('cuda'))
