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
    inputShape = (None, None, 3)
    batchSize = 8
    latentSize = 100
    img = load_img(os.path.join(os.path.dirname(__file__), '..','images', 'img.jpg'))
    img = np.array(img, dtype=np.float32) * (2/255) - 1
    img = np.array([img]*batchSize)
    new_img = load_img(os.path.join(os.getcwd(),'cropped', 'img.jpg'))
    new_img = np.array(new_img, dtype=np.float32) * (2/255) - 1
    new_img = np.array([new_img]*batchSize)
    encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=69)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer = 'adam', loss = 'mse')
    #rlrop = ReduceLROnPlateau(monitor = 'loss', factor=0.1, patience = 100)
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 50)
    # checkpoint
    #filepath = "best-model.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    #callbacks_list = [checkpoint]
    bvae.ae.fit(img, new_img,
                epochs=50,
                batch_size=batchSize,callbacks = [es])
    #bvae.ae.save('sr.h5')
    latentVec = bvae.encoder.predict(new_img)[0]
    pred = bvae.ae.predict(new_img)
    print(pred.shape)
    pred = np.uint8((pred + 1)* 255/2)
    print(pred.shape)
    """
    temp = new_img
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    pred *= 255
    pred[pred[:] > 255] = 255
    pred[pred[:] < 0] = 0
    pred = pred.astype(np.uint8)
    temp = shave(temp, 6)
    temp[:, :, ] = pre[:, :,0]
    output = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    ref = shave(ref.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)
    """
    pred = Image.fromarray(pred[0])
    pred.save("reconstructed_image.png")
    #cv2.imwrite("degraded.jpg",degraded)

if __name__ == "__main__":
    test()
