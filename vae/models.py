import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D, GaussianNoise)
from tensorflow.python.keras.models import Model
from utils import conv_block
from layers import SampleLayer


class vgg_encoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize = None,
                 latentSize=1000, latentConstraints = 'bvae', beta = 100., training = None):
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        inLayer = Input(self.inputShape, self.batchSize)
        x = conv_block(32, kernelSize = 3)(inLayer, training = self.training)
        x = MaxPool2D((2, 2), strides = (2, 2))(x)
        x = conv_block(64, kernelSize = 3)(x, training = self.training)
        x = MaxPool2D((2, 2), strides = (2, 2))(x)
        x = GaussianNoise(0.1)(x)
        x = conv_block(128, kernelSize = 3)(x, training = self.training)
        x = conv_block(64, kernelSize = 1)(x, training = self.training)
        x = conv_block(128, kernelSize = 3)(x, training = self.training)
        x = MaxPool2D((2, 2), strides = (2, 2))(x)
        x = conv_block(256, kernelSize = 3)(x, training = self.training)
        x = conv_block(128, kernelSize = 1)(x, training = self.training)
        x = GaussianNoise(0.1)(x)
        x = conv_block(256, kernelSize = 3)(x, training = self.training)
        x = MaxPool2D((2, 2), strides = (2, 2))(x)
        x = conv_block(512, kernelSize = 3)(x, training = self.training)
        x = conv_block(256, kernelSize = 1)(x, training = self.training)
        x = conv_block(512, kernelSize = 3)(x, training = self.training)
        x = GaussianNoise(0.1)(x)
        x = conv_block(256, kernelSize = 1)(x, training = self.training)
        x = conv_block(512, kernelSize = 3)(x, training = self.training)
        x = MaxPool2D((2, 2), strides=(2, 2))(x)
        x = GaussianNoise(0.1)(x)
        x = conv_block(1024, kernelSize = 3)(x, training = self.training)
        x = conv_block(512, kernelSize = 1)(x, training = self.training)
        x = conv_block(1024, kernelSize = 3)(x, training = self.training)
        x = conv_block(512, kernelSize = 1)(x, training = self.training)
        x = GaussianNoise(0.1)(x)
        x = conv_block(1024, kernelSize = 3)(x, training = self.training)
        mean = Conv2D(filters = self.latentSize, kernel_size = (1, 1),
                      padding = 'same')(x)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters = self.latentSize, kernel_size = (1, 1),
                        padding = 'same')(x)
        logvar = GlobalAveragePooling2D()(logvar)
        sample = layers(self.latentConstraints, self.beta)([mean, logvar], training = self.training)
        return Model(inputs = inLayer, outputs = sample)

class vgg_decoder(Architecture):
    def __init__(self, inputShape = (256, 256, 3), batchSize = None, latentSize = 1000, training = None):
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def build(self):
        inLayer = Input([self.latentSize], self.batchSize)
        x = Reshape((1, 1, self.latentSize))(inLayer)
        x = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(x)
        x = conv_block(1024, kernelSize = 3)(x, training = self.training)
        x = conv_block(512, kernelSize = 1)(x, training = self.training)
        x = conv_block(1024, kernelSize = 3)(x, training = self.training)
        x = conv_block(512, kernelSize = 1)(x, training = self.training)
        x = conv_block(1024, kernelSize = 3)(x, training = self.training)
        x = UpSampling2D((2, 2))(x)
        x = conv_block(512, kernelSize = 3)(x, training = self.training)
        x = conv_block(256, kernelSize = 1)(x, training = self.training)
        x = conv_block(512, kernelSize = 3)(x, training = self.training)
        x = conv_block(256, kernelSize = 1)(x, training = self.training)
        x = conv_block(512, kernelSize = 3)(x, training = self.training)
        x = UpSampling2D((2, 2))(x)
        x = conv_block(256, kernelSize = 3)(x, training = self.training)
        x = conv_block(128, kernelSize = 1)(x, training = self.training)
        x = conv_block(256, kernelSize = 3)(x, training = self.training)
        x = UpSampling2D((2, 2))(x)
        x = conv_block(128, kernelSize = 3)(x, training = self.training)
        x = conv_block(64, kernelSize = 1)(x, training = self.training)
        x = conv_block(128, kernelSize = 3)(x, training = self.training)
        x = UpSampling2D((2, 2))(x)
        x = conv_block(64, kernelSize = 3)(x, training = self.training)
        x = UpSampling2D((2, 2))(x)
        x = conv_block(32, kernelSize = 3)(x, training = self.training)
        x = conv_block(64, kernelSize = 1)(x, training = self.training)
        x = Conv2D(filters = self.inputShape[-1], kernel_size = (1, 1),
                      padding = 'same', activation = "tanh")(x)
        return Model(inLayer, x)

def model_summary():
    encoder = vgg_encoder()
    encoder.model.summary()
    decoder = vgg_decoder()
    decoder.model.summary()

if __name__ == '__main__':
    test()
