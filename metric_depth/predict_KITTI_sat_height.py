import os
import glob
from torchvision.transforms import Compose

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

import torch
import cv2
import torch.nn.functional as F
from run_grd_to_sky_forward import predict_sat_height
from forward_mapping_KITTI import forward_mapping_v2

to_pil = transforms.ToPILImage()
dates = ['2011_09_29', '2011_09_26', '2011_10_03', '2011_09_30', '2011_09_28']

camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                            [0.0000, 482.7076, 125.0034],
                            [0.0000, 0.0000, 1.0000]]],
                        dtype=torch.float32, requires_grad=False).to('cuda')
transform = transforms.Compose([
    transforms.Resize((256, 1024)),  # 调整图片尺寸
    transforms.ToTensor()  # 转换为张量
])
# transform = transforms.Compose([
#     transforms.Resize((128, 512)),  # 调整图片尺寸
#     transforms.ToTensor()  # 转换为张量
# ])
camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

def get_images_from_directory(root_dir, extensions=['.jpg', '.png', '.jpeg', '.bmp', '.gif'], grd_image_width=1024, grd_image_height=256, sat_width=101):
    
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
                          # os.makedirs(os.path.join(parent_dir, 'grd_forward_map'), exist_ok=True)
                          os.makedirs(os.path.join(parent_dir, 'grd_forward_map_v2'), exist_ok=True)
                          # os.makedirs(os.path.join(parent_dir, 'sat_height'), exist_ok=True)
                          # os.makedirs(os.path.join(parent_dir, 'grd_depth'), exist_ok=True)
                          depth_dir = os.path.join(parent_dir, 'grd_depth')
                          for grd_image in tqdm(image_files, desc=f'Processing {os.path.join(date, time)}', leave=True):
                              if grd_image.endswith(tuple(extensions)):
                                  depth = torch.load(os.path.join(depth_dir, grd_image.replace('.png', '_grd_depth.pt'))).to('cuda')
                                  grd_image_pil = Image.open(os.path.join(image_dir, grd_image))
                                  grd_image_tensor = transform(grd_image_pil).unsqueeze(0).to('cuda')
                                  # sat_height, project_grd_img = predict_sat_height(grd_image_tensor, camera_k, depth)
                                  forward_mapping_image = forward_mapping_v2(depth, grd_image_tensor, camera_k)
                                #   print(sat_height.shape, project_grd_img.shape)
                                #   torch.save(sat_height, os.path.join(parent_dir, 'sat_height', grd_image.replace('.png', '_sat_height.pt')))
                                #   project_grd_img = to_pil(project_grd_img.squeeze(0))
                                #   project_grd_img.save(os.path.join(parent_dir, 'grd_forward_map', grd_image))
                                  image_bev_show = to_pil(forward_mapping_image)
                                  image_bev_show.save(os.path.join(parent_dir, 'grd_forward_map_v2', grd_image))
    return image_files

# Example usage
root_directory = '/home/wangqw/video_dataset/KITTI/depth_data'  # Replace with your directory path
images = get_images_from_directory(root_directory)

for img in images:
    print(img)