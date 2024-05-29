import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import cv2
import random

IMAGE_SIZE = 44

# 입력 인자 image_filenames는 numpy array
class Celeb_Dataset(Sequence):
    # 객체 생성 인자로 들어온 값을 객체 내부 변수로 할당.
    def __init__(self, image_filenames, batch_size=32, augmentor=None, shuffle=False, rescale=False):
        '''
        파라미터 설명
        image_filenames: opencv로 image를 로드할 파일의 절대 경로들
        batch_size: __getitem__(self, index) 호출 시 마다 가져올 데이터 batch 건수
        augmentor: albumentations 객체
        shuffle: 학습 데이터의 경우 epoch 종료시마다 데이터를0 섞을지 여부
        '''

        # 객체 생성 인자로 들어온 값을 객체 내부 변수로 할당.
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.shuffle = shuffle
        self.rescale = rescale
        # train data의 경우
        if self.shuffle:
            # 객체 생성시에 한번 데이터를 섞음.
            self.on_epoch_end()

    # Sequence를 상속받은 Dataset은 batch_size 단위로 입력된 데이터를 처리함.
    # __len__()은 전체 데이터 건수가 주어졌을 때 batch_size 단위로 몇번 데이터를 반환하는지 나타남

    def __len__(self):
        # batch_size단위로 데이터를 몇번 가져와야하는지 계산하기 위해 전체 데이터 건수를 batch_size로 나누되, 정수로 정확히 나눠지지 않을 경우 1회를 더한다.
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        # batch_size 단위로 image_array, label_array 데이터를 가져와서 변환한 뒤 다시 반환함
        # 인자로 몇번째 batch 인지를 나타내는 index를 입력하면 해당 순서에 해당하는 batch_size 만큼의 데이타를 가공하여 반환
        # batch_size 갯수만큼 변환된 input_batch 과 target_batch

        # index는 몇번째 batch인지를 나타냄.
        # batch_size만큼 순차적으로 데이터를 가져오려면 array에서 index*self.batch_size:(index+1)*self.batch_size 만큼의 연속 데이터를 가져오면 됨
        image_name_batch = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]

        # 만일 객체 생성 인자로 albumentation으로 만든 augmentor가 주어진다면 아래와 같이 augmentor를 이용하여 image 변환
        # albumentations은 개별 image만 변환할 수 있으므로 batch_size만큼 할당된 image_name_batch를 한 건씩 iteration하면서 변환 수행.
        # image_batch 배열은 float32 로 설정.
        blur_input_batch = np.zeros((image_name_batch.shape[0], 176, 176, 3))
        model_input_bath = np.zeros((image_name_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
        target_batch = np.zeros((image_name_batch.shape[0], 176, 176, 3))

        # batch_size에 담긴 건수만큼 iteration 하면서 opencv image load -> image augmentation 변환(augmentor가 not None일 경우)-> image_batch에 담음.
        for image_index in range(image_name_batch.shape[0]):
            image = cv2.cvtColor(cv2.imread(image_name_batch[image_index]), cv2.COLOR_BGR2RGB)



            image1 = self.augmentor(image=image)['image']
            image1 = cv2.resize(image1, (176, 176))
            blur_input_batch[image_index] = image1

            image2 = self.augmentor(image=image)['image']
            image2 = cv2.resize(image2, (IMAGE_SIZE, IMAGE_SIZE))
            model_input_bath[image_index] = image2/255.0

            image3 = cv2.resize(image, (176, 176))
            target_batch[image_index] = image3

        return blur_input_batch, model_input_bath, target_batch

    # epoch가 한번 수행이 완료 될 때마다 모델의 fit()에서 호출됨.
    def on_epoch_end(self):
        if (self.shuffle):
            # print('epoch end')

            random.shuffle(self.image_filenames)
        else:
            pass



