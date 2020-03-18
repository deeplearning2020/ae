import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D, GaussianNoise)
from tensorflow.python.keras.models import Model
from model_utils import ConvBnLRelu, SelfAttention
from sample_layer import SampleLayer

class Architecture(object):
    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize
        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')

class Darknet19Encoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=None,
                 latentSize=1000, latentConstraints='bvae', beta=100., training=None):
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3)(inLayer, training=self.training) # 1
        net = SelfAttention(32)(net, training = self.training)
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = GaussianNoise(0.2)(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 3
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 6
        net = GaussianNoise(0.3)(net)
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = GaussianNoise(0.3)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 17
        net = GaussianNoise(0.2)(net)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        logvar = GlobalAveragePooling2D()(logvar)
        sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)
        return Model(inputs=inLayer, outputs=sample)

class Darknet19Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=None, latentSize=1000, training=None):
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        inLayer = Input([self.latentSize], self.batchSize)
        net = Reshape((1, 1, self.latentSize))(inLayer)
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2,2))(net)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2,2))(net)
        net = UpSampling2D((2,2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(32, kernelSize=3)(net, training=self.training)
        net = SelfAttention(32)(net, training = self.training)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same', activation="tanh")(net)
        return Model(inLayer, net)

def test():
    d19e = Darknet19Encoder()
    d19e.model.summary()
    d19d = Darknet19Decoder()
    d19d.model.summary()

if __name__ == '__main__':
    test()
