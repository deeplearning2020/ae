import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from keras.callbacks import EarlyStopping
#from keras.callbacks import LearningRateScheduler
#from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import os, math
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img
from models import Darknet19Encoder, Darknet19Decoder
from matplotlib import pyplot as plt

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):
        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model
        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img

def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def test():
    inputShape = (256, 256, 3)
    batchSize = 8
    latentSize = 400
    img = load_img(os.path.join(os.getcwd(),'images','img.bmp'),target_size=inputShape[:-1])
    print(img)
    img = np.array(img, dtype=np.float32) * (2/255) - 1
    img = np.array([img]*batchSize)
    new_img = load_img(os.path.join(os.getcwd(),'cropped','cropped.bmp'),target_size=inputShape[:-1])
    print(new_img)
    new_img = np.array(new_img, dtype=np.float32) * (2/255) - 1
    new_img = np.array([new_img]*batchSize)
    encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=69)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer = 'adam', loss = 'mse')
    #rlrop = ReduceLROnPlateau(monitor = 'loss', factor=0.1, patience = 100)
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 70)
    # checkpoint
    #filepath = "best-model.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    #callbacks_list = [checkpoint]
    history = bvae.ae.fit(img,new_img,
                epochs=2000,
                batch_size=batchSize,callbacks = [es])
    #bvae.ae.save('sr.h5')
    latentVec = bvae.encoder.predict(new_img)[0]
    pred = bvae.ae.predict(new_img)
    pred = np.uint8((pred + 1)* 255/2)
    pred = Image.fromarray(pred[0])
    pred.save("reconstructed_image.png")
    """
    plt.plot(history.history['loss'])
    plt.title('Reconstruction loss on a SET12 image sample')
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='lower right')
    plt.savefig("loss.pdf")
    """

if __name__ == "__main__":
    test()
