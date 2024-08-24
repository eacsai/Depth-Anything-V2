import os
import glob
from torchvision.transforms import Compose

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from tqdm import tqdm


import torch
import cv2
import torch.nn.functional as F


dates = ['2011_09_29', '2011_09_26', '2011_10_03', '2011_09_30', '2011_09_28']

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                        [0.0000, 482.7076, 125.0034],
                        [0.0000, 0.0000, 1.0000]]],
                    dtype=torch.float32, requires_grad=True).to('cuda')
camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

depth_v2_load_from = '/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth'

def get_images_from_directory(root_dir, extensions=['.jpg', '.png', '.jpeg', '.bmp', '.gif'], grd_image_width=1024, grd_image_height=256, sat_width=101):
    
    depth_anything_v1 = DepthAnything(model_configs['vitl'])
    depth_anything_v1.load_state_dict(torch.load(f'/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_vitl14.pth'))
    depth_anything_v1 = depth_anything_v1.to('cuda').eval()

    depth_anything_v2 = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 80})
    depth_anything_v2.load_state_dict(torch.load(depth_v2_load_from, map_location='cpu'))
    depth_anything_v2 = depth_anything_v2.to('cuda').eval()

    v, u = torch.meshgrid(torch.arange(0, grd_image_height, dtype=torch.float32),
                            torch.arange(0, grd_image_width, dtype=torch.float32))
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to('cuda')
    grd_image_files = []
    with torch.no_grad():
      for root, date_dirs, _ in tqdm(os.walk(root_dir), total=len(dates), desc="Processing directories"):
          for date in date_dirs:
              if date in dates:
                for _, time_dirs, _ in os.walk(os.path.join(root_dir, date)):
                    for time in tqdm(time_dirs, desc="Processing time_dirs", leave=False):
                      image_dir = os.path.join(root_dir, date, time, 'image_02/data')
                      for _, _, image_files in os.walk(image_dir):
                          parent_dir = os.path.abspath(os.path.join(image_dir, os.pardir))
                          os.makedirs(os.path.join(parent_dir, 'grd_depth'), exist_ok=True)
                          os.makedirs(os.path.join(parent_dir, 'sat_height'), exist_ok=True)
                          os.makedirs(os.path.join(parent_dir, 'grd_height'), exist_ok=True)
                          for grd_image in tqdm(image_files, desc=f'Processing {os.path.join(date, time)}', leave=True):
                              if grd_image.endswith(tuple(extensions)):
                                  grd_image_files.append(os.path.join(image_dir, grd_image))
                                  grd_image_cv2 = cv2.imread(os.path.join(image_dir, grd_image))
                                  
                                  grd_image_cv2_v1 = cv2.cvtColor(grd_image_cv2, cv2.COLOR_BGR2RGB) / 255.0
                                  grd_image_cv2_v1 = transform({'image': grd_image_cv2_v1})['image']
                                  grd_image_cv2_v1 = torch.from_numpy(grd_image_cv2_v1).unsqueeze(0).to('cuda')
                                  mask = depth_anything_v1(grd_image_cv2_v1)
                                  mask = F.interpolate(mask[None], (256, 1024), mode='bilinear', align_corners=False)[0, 0]
                                  mask[mask != 0] = 1

                                  depth = depth_anything_v2.infer_image(grd_image_cv2, 518)
                                  depth = torch.from_numpy(depth).to('cuda').unsqueeze(0)
                                  depth = F.interpolate(depth[None], (256, 1024), mode='bilinear', align_corners=False)[0, 0]
                                  depth = depth * mask

                                  torch.save(depth, os.path.join(parent_dir, 'grd_depth', f'{grd_image.replace(".png", "")}_grd_depth.pt'))

                                  xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]
                                  depth = depth.unsqueeze(0).unsqueeze(-1)
                                  # xyz_grd = xyz_w * depth / meter_per_pixel
                                  xyz_grd = xyz_w * depth * 1.2
                                  B, H, W, C = xyz_grd.shape
                                  xyz_grd = xyz_grd.view(B*H*W, -1)

                                  max_height = xyz_grd[:,1].max()
                                  xyz_grd[:,1] = max_height - xyz_grd[:,1]
                                  grd_height = xyz_grd[:,1].view(B, H, W, 1)
                                  torch.save(grd_height, os.path.join(parent_dir, 'grd_height', f'{grd_image.replace(".png", "")}_grd_height.pt'))
    return image_files

# Example usage
root_directory = '/home/wangqw/video_dataset/KITTI/depth_data'  # Replace with your directory path
images = get_images_from_directory(root_directory)

for img in images:
    print(img)