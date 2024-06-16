import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce
import os
import glob
import numpy as np
from dotenv import load_dotenv
load_dotenv()
def show_images(images, titles, ncols):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
    plt.show()
def crop_center_image(img_array):
    h, w, _ = img_array.shape
    print(h, w)
    cropped_img = img_array[int((h - w) / 2):int(-(h - w) / 2), :]
    return cropped_img
def resize_image(image, downscale=4):
    resized_image = pyramid_reduce(image, downscale=downscale, multichannel=True)
    return resized_image

base_path = os.getenv('BASE_PATH') + '/processed'


def clean_file_paths(directory_path):
    file_list = sorted(glob.glob(os.path.join(base_path, directory_path, '*.npy')))
    cleaned_list = [file_path.replace('\\', '/') for file_path in file_list]
    return cleaned_list


def convert_BGR_to_RGB(image):
    # 이미지의 BGR 채널을 RGB로 변환
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]

    # RGB 이미지로 결합
    rgb_image = np.stack((red, green, blue), axis=-1)

    return rgb_image
