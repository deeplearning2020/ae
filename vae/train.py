import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import os, math
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img
from models import Darknet19Encoder, Darknet19Decoder

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):
        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model
        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def test():
    inputShape = (256, 256, 3)
    batchSize = 8
    latentSize = 100
    print(os.getcwd())
    img = load_img(os.path.join(os.path.dirname(__file__), '..','images', 'img.jpg'), target_size=inputShape[:-1])
    img = np.array(img, dtype=np.float32) * (2/255) - 1
    img = np.array([img]*batchSize)
    print(os.getcwd())
    new_img = load_img(os.path.join(os.getcwd(),'cropped', 'img.jpg'), target_size=inputShape[:-1])
    new_img = np.array(new_img, dtype=np.float32) * (2/255) - 1
    new_img = np.array([new_img]*batchSize)
    encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=69)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer = 'adam', loss = 'mse')
    #rlrop = ReduceLROnPlateau(monitor = 'loss', factor=0.1, patience = 100)
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 50)
    bvae.ae.fit(img, new_img,
                epochs=5000,
                batch_size=batchSize,callbacks = [es])
    latentVec = bvae.encoder.predict(new_img)[0]
    pred = bvae.ae.predict(new_img)
    pred = np.uint8((pred + 1)* 255/2)
    print(pred.shape)
    pred = Image.fromarray(pred[0])
    pred.save("reconstructed_image.png")
    pred.show()

if __name__ == "__main__":
    test()
