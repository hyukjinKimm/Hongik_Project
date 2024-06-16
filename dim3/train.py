from dotenv import load_dotenv
import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 상위 폴더 경로
parent_dir = os.path.dirname(current_dir)
# sys.path에 상위 폴더 경로 추가
sys.path.append(parent_dir)
from espcn_3dim import create_espcn_3dim_model
from dataGen import Celeb_Dataset
import albumentations as A
from callbacks import callbacks
from criteria import dim3_psnr
import tensorflow.python.keras as tf_keras
from keras import __version__
import matplotlib.pyplot as plt
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
train_df = data_df[data_df['partition']=='0']
test_df = data_df[data_df['partition']=='1']
val_df = data_df[data_df['partition']=='2']
train_image_filenames = train_df['path'].values
validation_image_filenames = val_df['path'].values

print(train_df)

celeb_augmentor = A.Compose([
    A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(3, 3), p=1.0)  # 가우시안 블러 적용
])
#
# Sequence를 상속받은 Celeb_Dataset을 image 파일 위치,  albumentations 변환 객체를 입력하여 생성.
tr_ds = Celeb_Dataset(train_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True)
val_ds = Celeb_Dataset(validation_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=False, rescale=True)
# 배치 데이터를 가져오기 위해 iter와 next 사용
tr_iter = iter(tr_ds)
input_batch, target_batch = next(tr_iter)

# 필요한 개수만큼 이미지를 출력합니다 (여기서는 배치에서 5개의 이미지만 출력)
num_images_to_display = 5


# 함수 정의: 이미지를 출력하는 함수
def display_images(input_images, target_images, num_images):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(input_images[i])
        plt.title("Input")
        height, width, _ = input_images[i].shape
        plt.xlabel(f"{width}x{height}")
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(target_images[i])
        plt.title("Target")
        height, width, _ = target_images[i].shape
        plt.xlabel(f"{width}x{height}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 배치에서 첫 num_images_to_display 개수만큼 이미지를 출력
# display_images(input_batch, target_batch, num_images_to_display)

model = create_espcn_3dim_model(SCALE)
model.summary()
#
model.compile(optimizer='adam', loss='mse', metrics=[dim3_psnr.dim3_psnr])
# 모델 훈련
history = model.fit(tr_ds, batch_size=BATCH_SIZE, epochs=20, shuffle=True,
                    validation_data=val_ds,
                    callbacks=[callbacks.mcp_cb, callbacks.rlr_cb, callbacks.ely_cb])

# import pickle
#
# # history 객체 저장
# save_path = os.getenv('HISTORY_PATH') + '\history.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(history.history, f)