from dotenv import load_dotenv
import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더 경로
parent_dir = os.path.dirname(current_dir)

# sys.path에 상위 폴더 경로 추가
sys.path.append(parent_dir)

from espcn_1dim import create_espcn_1dim_model
from dataGen import Celeb_Dataset_1dim
import albumentations as A
from callbacks import callbacks
from criteria import dim1_psnr
import tensorflow.python.keras as tf_keras
from keras import __version__

tf_keras.__version__ = __version__

print(tf_keras.__version__)
load_dotenv()

base_path = os.getenv('BASE_PATH')
img_base_path = os.getenv('IMAGE_BASE_PATH')

df = pd.read_csv(os.path.join(base_path + '/img_align_celeba/list_eval_partition.csv'))
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


celeb_augmentor = A.Compose([
    #A.Blur(blur_limit=(7, 7))  # 블러 적용, blur_limit을 (7, 7)로 설정하여 강도를 7로 고정합니다.
    A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(3, 3), p=1.0)  # 가우시안 블러 적용
])



# Sequence를 상속받은 celeba Dataset을 image 파일 위치, label값, albumentations 변환 객체를 입력하여 생성.
tr_ds_r = Celeb_Dataset_1dim(train_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True, channel=0)
val_ds_r = Celeb_Dataset_1dim(validation_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True, channel=0)

tr_ds_g = Celeb_Dataset_1dim(train_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True, channel=1)
val_ds_g = Celeb_Dataset_1dim(validation_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True, channel=1)

tr_ds_b = Celeb_Dataset_1dim(train_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True, channel=2)
val_ds_b = Celeb_Dataset_1dim(validation_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True, channel=2)


model_R = create_espcn_1dim_model(SCALE)
model_G = create_espcn_1dim_model(SCALE)
model_B = create_espcn_1dim_model(SCALE)


model_R.compile(optimizer='adam', loss='mse', metrics=[dim1_psnr.dim1_psnr])
model_G.compile(optimizer='adam', loss='mse', metrics=[dim1_psnr.dim1_psnr])
model_B.compile(optimizer='adam', loss='mse', metrics=[dim1_psnr.dim1_psnr])
# 모델 훈련
history_R = model_R.fit(tr_ds_r, batch_size=BATCH_SIZE, epochs=20, shuffle=True,
                    validation_data=val_ds_r,
                    callbacks=[callbacks.mcp_cb_R, callbacks.rlr_cb, callbacks.ely_cb])
history_G = model_G.fit(tr_ds_g, batch_size=BATCH_SIZE, epochs=20, shuffle=True,
                    validation_data=val_ds_g,
                    callbacks=[callbacks.mcp_cb_G, callbacks.rlr_cb, callbacks.ely_cb])
history_B = model_B.fit(tr_ds_b, batch_size=BATCH_SIZE, epochs=20, shuffle=True,
                    validation_data=val_ds_b,
                    callbacks=[callbacks.mcp_cb_B, callbacks.rlr_cb, callbacks.ely_cb])
# import pickle
# # history 객체 저장
# save_path = os.getenv('HISTORY_PATH') + '\R\history.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(history_R.history, f)
# save_path = os.getenv('HISTORY_PATH') + '\G\history.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(history_G.history, f)
# save_path = os.getenv('HISTORY_PATH') + '\B\history.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(history_B.history, f)