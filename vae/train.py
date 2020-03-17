import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from keras.callbacks import EarlyStopping
import os
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

def test():
    inputShape = (256, 256, 3)
    batchSize = 8
    latentSize = 100
    img = load_img(os.path.join(os.path.dirname(__file__), '..','images', 'img.jpg'), target_size=inputShape[:-1])
    img.show()
    img = np.array(img, dtype=np.float32) * (2/255) - 1
    img = np.array([img]*batchSize)
    encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=69)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer = 'adam', loss = 'mse')
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1,patience = 50)
    bvae.ae.fit(img, img,
                epochs=5000,
                batch_size=batchSize,callbacks = [es])
    latentVec = bvae.encoder.predict(img)[0]
    pred = bvae.ae.predict(img)
    pred = np.uint8((pred + 1)* 255/2)
    pred = Image.fromarray(pred[0])
    pred.save("reconstruced_image.png")
    pred.show()

if __name__ == "__main__":
    test()
