import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Layer,  Conv2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.training import Model

class PixelShuffle(Layer):
    def __init__(self, scale_factor, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale_factor)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.scale_factor, input_shape[2] * self.scale_factor)


def create_espcn_3dim_model(scale_factor):
    # 입력 이미지의 형태를 정의합니다.
    input_shape = (44, 44, 3)  # RGB 이미지를 입력으로 받습니다.

    # 첫 번째 단계: 특징 추출 (Feature Extraction)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (5, 5), padding='same')(inputs)
    x = Activation('tanh')(x)

    # 두 번째 단계: 특징 추출 (Feature Extraction)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('tanh')(x)

    # 세 번째 단계: 업스케일링 (Upscaling)
    x = Conv2D(3 * (scale_factor ** 2), (3, 3), padding='same')(x)
    outputs = PixelShuffle(scale_factor)(x)

    # 모델을 생성합니다.
    model = Model(inputs=inputs, outputs=outputs)
    return model