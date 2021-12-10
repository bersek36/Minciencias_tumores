#import os
#import sys
#import numpy as np
import matplotlib.pyplot as plt
#import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Conv3DTranspose
from tensorflow.keras.layers import Dropout, Input, BatchNormalization, Concatenate
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.pooling import AveragePooling3D


def dice_coeff(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    y_pred = tf.where(K.greater_equal(y_pred,0.5),1.,0.)
    # y_pred = y_pred.astype(np.float32) 
    dice_numerator = K.sum(2 * y_true * y_pred, axis= axis) + epsilon
    dice_denominator = K.sum(y_true,axis= axis) + K.sum(y_pred, axis = axis) + epsilon
    dice_coefficient = K.mean(dice_numerator / dice_denominator,axis = 0)
    return dice_coefficient

def unet_3D(x,y,z):
    a = 32
    b = a*2
    c = b*2
    d = c*2
    e = d*2
    norm = 0.9
    #f = [16, 32, 64, 128, 256]
    inputs  = Input((x,y,z,1))
    
    #IMAGEN DE ENTRADA
    p0 = inputs
    print(p0.shape,"\n")

    #PRIMERA CONVOLUCION  
    c1 = Conv3D(a, kernel_size=(3,3,3), padding="same", strides=[1,1,1], activation="relu")(p0)
    c1 = BatchNormalization(momentum=norm)(c1)
    print(c1.shape)
    c2 = Conv3D(a, kernel_size=(3,3,3), padding="same", strides=[1,1,1], activation="relu")(c1)
    c2 = BatchNormalization(momentum=norm)(c2)
    print(c2.shape)
    p1 = MaxPool3D((2,2,2))(c2)
    print(p1.shape)

    #SEGUNDA CONVOLUCION
    c3 = Conv3D(b, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(p1)
    print(c3.shape)
    c3 = BatchNormalization(momentum=norm)(c3)
    c4 = Conv3D(b, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(c3)
    c4 = BatchNormalization(momentum=norm)(c4)
    print(c4.shape)
    p2 = MaxPool3D((2,2,2))(c4)
    print(p2.shape)

    #TERCERA CONVOLUCION
    c5 = Conv3D(c, kernel_size=(3, 3,3), padding="same", strides=[1,1,1], activation="relu")(p2)
    c5 = BatchNormalization(momentum=norm)(c5)
    print(c5.shape)
    c6 = Conv3D(c, kernel_size=(3, 3,3), padding="same", strides=[1,1,1], activation="relu")(c5)
    c6 = BatchNormalization(momentum=norm)(c6)
    print(c6.shape)
    p3 = MaxPool3D((2,2,2))(c6)
    print(p3.shape)
    
    #CUARTA CONVOLUCION
    c7 = Conv3D(d, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(p3)
    c7 = BatchNormalization(momentum=norm)(c7)
    print(c7.shape)
    c8 = Conv3D(d, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(c7)
    c8 = BatchNormalization(momentum=norm)(c8)
    print(c8.shape, " here go1 \n")
    p4 = MaxPool3D((2,2,2))(c8)
    print(p4.shape," here go\n ")

    
    #QUINTA CONVOLUCION SIN POOLING
    c9 = Conv3D(e, kernel_size=(3, 3,3), padding="same", strides=[1,1,1], activation="relu")(p4)
    c9 = BatchNormalization(momentum=norm)(c9)
    print(c9.shape)
    c10 = Conv3D(e, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(c9)
    c10 = BatchNormalization(momentum=norm)(c10)
    print(c10.shape,"\n")

        #PRIMERA DECONVOLUCION
    us1 = Conv3DTranspose(d ,(3,3,3),strides=(2,2,2), padding='same')(c10)
    print(us1.shape)
    concat1 = Concatenate(axis=-1)([us1,c8])
    print(concat1.shape)
    c11 = Conv3D(d, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(concat1)
    c11 = BatchNormalization(momentum=norm)(c11)
    print(c11.shape)
    c12 = Conv3D(d, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(c11)
    c12 = BatchNormalization(momentum=norm)(c12)
    print(c12.shape)

    
    #SEGUNDA DECONVOLUCION
    us2 = Conv3DTranspose(c ,(2,2,2),strides=(2,2,2),padding='same')(c12)
    print(us2.shape)
    concat2 = Concatenate(axis=-1)([us2,c6])
    print(concat2.shape)
    c13 = Conv3D(c, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(concat2)
    c13 = BatchNormalization(momentum=norm)(c13)
    print(c13.shape)
    c14 = Conv3D(c, kernel_size=(3, 3,3), padding="same", strides=[1,1,1], activation="relu")(c13)
    c14 = BatchNormalization(momentum=norm)(c14)
    print(c14.shape)

    
    #TERCERA DECONVOLUCION
    us3 = Conv3DTranspose(b ,(2,2,2), strides=(2,2,2),padding='same')(c14)
    print(us3.shape)
    concat3 = Concatenate(axis=-1)([us3,c4])
    print(concat3.shape)
    c15 = Conv3D(b, kernel_size=(3, 3,3), padding="same", strides=[1,1,1], activation="relu")(concat3)
    c15 = BatchNormalization(momentum=norm)(c15)
    print(c15.shape)
    c16 = Conv3D(b, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(c15)
    c16 = BatchNormalization(momentum=norm)(c16)
    print(c16.shape)
    
    #CUARTA DECONVOLUCION
    us4 = Conv3DTranspose(a ,(2,2,2), strides=(2,2,2),padding='same')(c16)
    print(us4.shape)
    concat4 = Concatenate(axis=-1)([us4,c2])
    print(concat4.shape)
    c17 = Conv3D(a, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(concat4)
    c17 = BatchNormalization(momentum=norm)(c17)
    print(c17.shape)
    c18 = Conv3D(a, kernel_size=(3,3, 3), padding="same", strides=[1,1,1], activation="relu")(c17)
    c18 = BatchNormalization(momentum=norm)(c18)
    print(c18.shape,"\n")

   
    
    #PREDICCION
    outputs = Conv3D(1, (1,1, 1), padding="same", activation="sigmoid")(c18)
    print(outputs.shape)

    model = Model(inputs, outputs)
    return model

'''
Plotear imagen original, mascara y prediccion
'''
def plot_slice(prueba_img, test_mask_img, predictions_img):
    fig = plt.figure(figsize=(16,16))
    fig.subplots_adjust(hspace =1 ,wspace =1)

    ax1 = fig.add_subplot(1,3,1)
    ax1.title.set_text('imagen')
    ax1.axis("off")
    ax1.imshow(prueba_img[:,:,8],cmap="gray")

    ax2 = fig.add_subplot(1,3,2)
    ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(test_mask_img[:,:,8],cmap="gray")

    ax3 = fig.add_subplot(1,3,3)
    ax3.title.set_text('prediccion')
    ax3.axis("off")
    ax3.imshow(predictions_img[:,:,8]>0.9,cmap="gray")

    plt.show()


