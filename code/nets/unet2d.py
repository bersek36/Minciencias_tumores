import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

'''
CONSTRUCCION RED UNET2D
'''
def unet2D(image_row,image_col):
    a = 32
    b = a*2
    c = b*2
    d = c*2
    e = d*2
    norm = 0.9
    #f = [16, 32, 64, 128, 256]
    inputs = Input((image_row,image_col,1)) 
    
    # IMAGEN DE ENTRADA
    p0 = inputs

    # PRIMERA CONVOLUCION  
    c1 = Conv2D(a, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p0)
    c1 = BatchNormalization(momentum=norm)(c1)
    c2 = Conv2D(a, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c1)
    c2 = BatchNormalization(momentum=norm)(c2)
    p1 = MaxPool2D((2,2))(c2)
    
    # SEGUNDA CONVOLUCION
    c3 = Conv2D(b, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p1)
    c3 = BatchNormalization(momentum=norm)(c3)
    c4 = Conv2D(b, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c3)
    c4 = BatchNormalization(momentum=norm)(c4)
    p2 = MaxPool2D((2,2))(c4)
    
    # TERCERA CONVOLUCION
    c5 = Conv2D(c, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p2)
    c5 = BatchNormalization(momentum=norm)(c5)
    c6 = Conv2D(c, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c5)
    c6 = BatchNormalization(momentum=norm)(c6)
    p3 = MaxPool2D((2,2))(c6)
    
    # CUARTA CONVOLUCION
    c7 = Conv2D(d, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p3)
    c7 = BatchNormalization(momentum=norm)(c7)
    c8 = Conv2D(d, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c7)
    c8 = BatchNormalization(momentum=norm)(c8)
    p4 = MaxPool2D((2,2))(c8)
    
    # QUINTA CONVOLUCION SIN POOLING
    c9 = Conv2D(e, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p4)
    c9 = BatchNormalization(momentum=norm)(c9)
    c10 = Conv2D(e, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c9)
    c10 = BatchNormalization(momentum=norm)(c10)

    # PRIMERA DECONVOLUCION
    us1 = Conv2DTranspose(d ,(2,2),strides=(2, 2), padding='same')(c10)
    concat1 = Concatenate()([us1,c8])
    c11 = Conv2D(d, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat1)
    c11 = BatchNormalization(momentum=norm)(c11)
    c12 = Conv2D(d, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c11)
    c12 = BatchNormalization(momentum=norm)(c12)

    # SEGUNDA DECONVOLUCION
    us2 = Conv2DTranspose(c ,(2,2),strides=(2, 2), padding='same')(c12)
    concat2 = Concatenate()([us2,c6])
    c13 = Conv2D(c, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat2)
    c13 = BatchNormalization(momentum=norm)(c13)
    c14 = Conv2D(c, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c13)
    c14 = BatchNormalization(momentum=norm)(c14)

    # TERCERA DECONVOLUCION
    us3 = Conv2DTranspose(b ,(2,2),strides=(2, 2), padding='same')(c14)
    concat3 = Concatenate()([us3,c4])
    c15 = Conv2D(b, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat3)
    c15 = BatchNormalization(momentum=norm)(c15)
    c16 = Conv2D(b, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c15)
    c16 = BatchNormalization(momentum=norm)(c16)

    # CUARTA DECONVOLUCION
    us4 = Conv2DTranspose(a ,(2,2),strides=(2, 2), padding='same')(c16)
    concat4 = Concatenate()([us4,c2])
    c17 = Conv2D(a, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat4)
    c17 = BatchNormalization(momentum=norm)(c17)
    c18 = Conv2D(a, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c17)
    c18 = BatchNormalization(momentum=norm)(c18)
    
    # SALIDAS
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(c18)

    model = Model(inputs, outputs)

    return model

'''
METRICA
'''
def dice_coeff(y_true, y_pred, axis=(1, 2), epsilon=0.00001):
    y_pred = tf.where(K.greater_equal(y_pred,0.5),1.,0.)
    dice_numerator = K.sum(2 * y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true,axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean(dice_numerator/dice_denominator, axis=0)
    return dice_coefficient


'''
Plotear imagen original, mascara y prediccion
'''
def plot_slice(prueba_img, test_mask_img, predictions_img):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.title.set_text('imagen')
    ax1.axis("off")
    ax1.imshow(prueba_img, cmap="gray")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(test_mask_img, cmap="gray")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.title.set_text('prediccion')
    ax3.axis("off")
    ax3.imshow(predictions_img>0.9, cmap="gray")

    plt.show()