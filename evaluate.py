from utils.utils import *
from model import create_model
from keras.layers import Input
from skimage.transform import pyramid_expand
from dotenv import load_dotenv
import os
load_dotenv()
x_test_list = clean_file_paths('x_test')
y_test_list = clean_file_paths('y_test')
print(len(x_test_list), len(y_test_list))
print(x_test_list[0])

test_idx = 177


upscale_factor = 4
inputs = Input(shape=(44, 44, 3))
model = create_model(inputs, upscale_factor)

WEIGHT_PATH = os.getenv('WEIGHT_FOLTER_PATH') + '/weight.08-0.0013019238.weights.h5'
model.load_weights(WEIGHT_PATH)

x1_test = np.load(x_test_list[test_idx])
x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)
y1_test = np.load(y_test_list[test_idx])
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

print(x1_test.shape, y1_test.shape)

x1_test = (x1_test * 255).astype(np.uint8)
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

x1_test = convert_BGR_to_RGB(x1_test)
x1_test_resized = convert_BGR_to_RGB(x1_test_resized)
y1_test = convert_BGR_to_RGB(y1_test)
y_pred = convert_BGR_to_RGB(y_pred)

show_images([x1_test,  x1_test_resized,y_pred, y1_test], ['input',  'resized','output', 'groundtruth'], ncols=4)

