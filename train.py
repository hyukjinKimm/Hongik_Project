import os
import pickle
from keras.layers import Input
from DataGenerator import DataGenerator
from callbacks.callbacks import *
from dotenv import load_dotenv
from model import create_model
from utils.utils import *
load_dotenv()
x_train_list = clean_file_paths('x_train')
x_val_list = clean_file_paths('x_val')

print(len(x_train_list), len(x_val_list))
print(x_train_list[0])
print(x_val_list[0])

train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=True)
val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=False)
upscale_factor = 4
inputs = Input(shape=(44, 44, 3))

model = create_model(inputs, upscale_factor)
history = model.fit_generator(train_gen, validation_data=val_gen, epochs=10, verbose=1, callbacks=[mcp_cb])

with open(os.getenv('HISTORY_PATH') + '/history.pickle', 'wb') as file:
    pickle.dump(history.history, file)