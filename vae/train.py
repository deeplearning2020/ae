import os, math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import load_img
from models import vgg_encoder, vgg_decoder
from matplotlib import pyplot as plt

class variational_autoencoder(object):
    def __init__(self, encoder_architecture, 
                 decoder_architecture):
        self.encoder = encoder_architecture.model
        self.decoder = decoder_architecture.model
        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))


def main():

    inputShape = (256, 256, 3)
    batchSize = 8
    latentSize = 400

    hr_image = load_img(os.path.join(os.getcwd(),'hr_image','HR.bmp'),target_size=inputShape[:-1]) ## loading the high-resolution image
    hr_image = np.array(hr_image, dtype = np.float32) * (2/255) - 1
    hr_image = np.array([hr_image]*batchSize) ## creating fake batches


    lr_image = load_img(os.path.join(os.getcwd(),'lr_image','LR.bmp'),target_size=inputShape[:-1]) ## loading the low-resolution image
    lr_image = np.array(lr_image, dtype = np.float32) * (2/255) - 1
    lr_image = np.array([lr_image]*batchSize) ## creating fake batches



    encoder = vgg_encoder(inputShape, latentSize = latentSize, latentConstraints='bvae', beta=69)

    decoder = vgg_decoder(inputShape, latentSize = latentSize)

    bvae = variational_autoencoder(encoder, decoder)
    
    bvae.ae.compile(optimizer = 'adam', loss = 'mse')
    
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 70) ## early stopping to prevent overfitting

    history = bvae.ae.fit(lr_image, hr_image,
                epochs = 2000,
                batch_size = batchSize, callbacks = [es])

    """ reconstrucing high-resolution image from the low-resolution image """
    pred = bvae.ae.predict(lr_image)
    pred = np.uint8((pred + 1)* 255/2)
    pred = Image.fromarray(pred[0])
    pred.save("reconstructed_HR_image.png")
    
    """ plotting training loss """
    plt.plot(history.history['loss'])
    plt.title('Reconstruction loss on a SET12 image sample')
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='lower right')
    plt.savefig("loss.pdf")

if __name__ == "__main__":
    main()
