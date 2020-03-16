import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D)
from tensorflow.python.keras.models import Model
from model_utils import ConvBnLRelu
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
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 3
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training) # 4
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 5
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 6
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training) # 7
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 8
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 9
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 10
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 11
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 12
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 14
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 15
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 16
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 17
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 18
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
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training)
        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(32, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same', activation="tanh")(net)
        return Model(inLayer, net)

class Darknet53Encoder(Architecture):
    def __init__(self, inputShape=(None, None, None, None), name='darkent53_encoder'):
        super().__init__(inputShape, name)
        self.Build()

    def Build(self):
        raise NotImplementedError('this architecture is not complete')

    def ConvBlock(self):
        raise NotImplementedError('this architecture is not complete')

def test():
    d19e = Darknet19Encoder()
    d19e.model.summary()
    d19d = Darknet19Decoder()
    d19d.model.summary()

if __name__ == '__main__':
    test()
