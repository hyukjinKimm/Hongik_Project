from tensorflow.python.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint
from dotenv import load_dotenv
import os
load_dotenv()

WEIGHT_PATH = os.getenv('WEIGHT_PATH')
FILEPATH = WEIGHT_PATH + '\weight.{epoch:02d}-{val_loss:.10f}.weights.h5'
# validation loss가 향상되는 모델만 저장.
mcp_cb = ModelCheckpoint(filepath= FILEPATH, monitor='val_loss',
                         save_best_only=True, save_weights_only=True, mode='min', verbose=1)
# 5번 iteration내에 validation loss가 향상되지 않으면 learning rate을 기존 learning rate * 0.2로 줄임.
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min', verbose=1)
# 10번 iteration내에 validation loss가 향상되지 않으면 더 이상 학습하지 않고 종료
ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)