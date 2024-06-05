import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from utils.utils import *

load_dotenv()
base_path = os.getenv('BASE_PATH')
img_base_path = os.path.join(base_path, 'img_align_celeba').replace('\\', '/')
target_img_path = os.path.join(base_path, 'processed').replace('\\', '/')
eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)
img_path = os.path.join(img_base_path, eval_list[7][0])
img_sample = Image.open(img_path)
img_sample = np.array(img_sample)
crop_sample = crop_center_image(img_sample)
resized_sample = resize_image(crop_sample, downscale=4)

show_images([img_sample, crop_sample,resized_sample], ['Original Image', 'Cropped Image', 'Resized Image'], ncols=3)

#
# downscale = 4
# n_train = 162770
# n_val = 19867
# n_test = 19962
#
# for i, e in enumerate(eval_list):
#     if (i%1000) == 0:
#         print(i)
#     filename, ext = os.path.splitext(e[0])
#
#     img_path = os.path.join(img_base_path, e[0])
#
#     img = Image.open(img_path)
#     img = np.array(img)
#     crop = crop_center_image(img)
#     resized = resize_image(crop, downscale=downscale)
#     norm = (crop.astype(np.float64) - crop.min()) / (crop.max() - crop.min())
#     if int(e[1]) == 0:
#         np.save(os.path.join(target_img_path, 'x_train', filename + '.npy').replace('\\', '/'), resized)
#         np.save(os.path.join(target_img_path, 'y_train', filename + '.npy').replace('\\', '/'), norm)
#     elif int(e[1]) == 1:
#         np.save(os.path.join(target_img_path, 'x_val', filename + '.npy').replace('\\', '/'), resized)
#         np.save(os.path.join(target_img_path, 'y_val', filename + '.npy').replace('\\', '/'), norm)
#     elif int(e[1]) == 2:
#         np.save(os.path.join(target_img_path, 'x_test', filename + '.npy').replace('\\', '/'), resized)
#         np.save(os.path.join(target_img_path, 'y_test', filename + '.npy').replace('\\', '/'), norm)
#
