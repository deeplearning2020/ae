import tensorflow as tf
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose, 
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, 
            Reshape, GlobalAveragePooling2D, Layer)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.initializers import RandomNormal,glorot_uniform

init = glorot_uniform()
class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    def __call__(self, net, training=None):
        net = Conv2D(self.filters, self.kernelSize,kernel_initializer = init,strides=self.strides, padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        return net
class DeconvRelu(object):
    def __init__(self, filters, kernelSize, strides=2):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    def __call__(self, net, training=None):
        net = Conv2DTranspose(self.filters, self.kernelSize,kernel_initializer = init,strides=self.strides, padding='same')(net)
        net = LeakyReLU()(net)
        return net

class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer=init, trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer=init,
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer=init,
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer=init,
                                        name='kernel_h',
                                        trainable=True)
        super(SelfAttention, self).build(input_shape)
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]
        beta = K.softmax(s, axis=-1)  # attention map
        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

