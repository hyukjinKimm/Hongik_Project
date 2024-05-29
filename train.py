from dotenv import load_dotenv
import pandas as pd
import os
from dataGen import Celeb_Dataset
import albumentations as A
from espcn_3dim import create_espcn_3dim_model
from callbacks import callbacks
from criteria import dim3_psnr
import tensorflow.python.keras as tf_keras
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
train_df = data_df[data_df['partition']=='0']
test_df = data_df[data_df['partition']=='1']
val_df = data_df[data_df['partition']=='2']
train_image_filenames = train_df['path'].values
validation_image_filenames = val_df['path'].values


celeb_augmentor = A.Compose([
    A.Blur(blur_limit=(7, 7))  # 블러 적용, blur_limit을 (7, 7)로 설정하여 강도를 7로 고정합니다.
])

# Sequence를 상속받은 CnD_Dataset을 image 파일 위치, label값, albumentations 변환 객체를 입력하여 생성.
tr_ds = Celeb_Dataset(train_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True)
val_ds = Celeb_Dataset(validation_image_filenames, batch_size=BATCH_SIZE, augmentor=celeb_augmentor, shuffle=True, rescale=True)


model = create_espcn_3dim_model(SCALE)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=[dim3_psnr.dim3_psnr])
# 모델 훈련
history = model.fit(tr_ds, batch_size=BATCH_SIZE, epochs=2, shuffle=True,
                    validation_data=val_ds,
                    callbacks=[callbacks.mcp_cb, callbacks.rlr_cb, callbacks.ely_cb])

import pickle

# history 객체 저장
save_path = os.getenv('HISTORY_PATH') + '\history.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(history.history, f)