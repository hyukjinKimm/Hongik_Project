from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os
from testDataGen import Celeb_Dataset
import albumentations as A
from espcn_1dim import create_espcn_1dim_model
import tensorflow.python.keras as tf_keras
import matplotlib.pyplot as plt
from keras import __version__
tf_keras.__version__ = __version__
print(tf_keras.__version__)
load_dotenv()

base_path = os.getenv('BASE_PATH')
img_base_path = os.getenv('IMAGE_BASE_PATH')

df = pd.read_csv(os.path.join(base_path + '/img_align_celeba/list_eval_partition.csv'))
# Train 0, Validation: 1, Test: 2
paths = []
for filename in os.listdir(img_base_path):
    if '.jpg' in filename:
        file_path = img_base_path + '/' + filename
        paths.append(file_path)
print('이미지 갯수: ', len(paths))
pd.set_option('display.max_colwidth', 200)
data_df = pd.DataFrame({ 'path':paths, 'partition':df.iloc[:,1].astype('str')})


BATCH_SIZE = 32
SCALE = 4

# 전체 데이터 세트에서 학습과 테스트용 메타 정보 DataFrame 생성.

test_df = data_df[data_df['partition']=='2']
test_image_filenames = test_df['path'].values

celeb_augmentor = A.Compose([
   # A.Blur(blur_limit=(7, 7))  # 블러 적용, blur_limit을 (7, 7)로 설정하여 강도를 7로 고정합니다.
    A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(3, 3), p=1.0)  # 가우시안 블러 적용
])

# Sequence를 상속받은 CnD_Dataset을 image 파일 위치, label값, albumentations 변환 객체를 입력하여 생성.
test_ds = Celeb_Dataset(test_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True)


model_R = create_espcn_1dim_model(SCALE)
model_G = create_espcn_1dim_model(SCALE)
model_B = create_espcn_1dim_model(SCALE)

weights_R_save_path = os.getenv('WEIGHT_PATH') + '\dim1\R\weight.01-0.0022197319.weights.h5'
weights_G_save_path = os.getenv('WEIGHT_PATH') + '\dim1\G\weight.01-0.0022105533.weights.h5'
weights_B_save_path = os.getenv('WEIGHT_PATH') + '\dim1\B\weight.01-0.0021295890.weights.h5'
model_R.load_weights(weights_R_save_path)
model_G.load_weights(weights_G_save_path)
model_B.load_weights(weights_B_save_path)


blur_input_batch, input_batch, target_batch = next(iter(test_ds))

predict_R = model_R.predict(np.expand_dims(input_batch[0:,:,:,0], axis=3))
predict_G = model_G.predict(np.expand_dims(input_batch[0:,:,:,1], axis=3))
predict_B = model_B.predict(np.expand_dims(input_batch[0:,:,:,2], axis=3))

print(predict_R[0].shape)
print(predict_G[0].shape)
print(predict_B[0].shape)

img1 = np.array(blur_input_batch[0], dtype='int32')

img2 = np.concatenate((predict_R[0], predict_G[0], predict_B[0]), axis=-1)

img3 = np.array(target_batch[0], dtype='int32')

print(img2.shape)
images = [img1, img2, img3]
titles = ['Blur', 'Predict', 'Target']


def display_images_with_titles(images, titles):
    """
    메모리에 있는 이미지들을 화면에 출력합니다.

    :param images: 이미지 배열 리스트 (각 이미지 배열은 (height, width, channels) 형식)
    :param titles: 이미지 제목 리스트
    """
    if len(images) != len(titles):
        raise ValueError("이미지 배열과 제목 배열의 길이가 같아야 합니다.")

    # 이미지 읽기 및 출력
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))  # 이미지 개수에 맞게 서브플롯 생성

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')  # 축 제거

        # 이미지의 픽셀 값을 출력
        ax.text(0, -10, f"Pixel range: [{img.min()}, {img.max()}]", color='white', fontsize=8)

    plt.tight_layout()
    plt.show()

display_images_with_titles(images, titles)
