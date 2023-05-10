import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import ELU, LeakyReLU, Add
from tensorflow.keras.applications.resnet import ResNet101 as pretrained_ResNet101

import os
import sys
import random
import cv2
import matplotlib.pyplot as plt
import time

## FUNCTIONS

def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


    return (1 - intersection /(np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))


def voe_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (1 - intersection /
            (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
   
## LAYERS

   

def expend_as(x, n):
    y = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
               arguments={'repnum': n})(x)

    return y


# def conv_bn_act(x, filters, drop_out=0.0):
#     x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

#     if drop_out > 0:
#         x = Dropout(drop_out)(x)

#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     return x

def conv_bn_act(x, filters, kernel_size=3, drop_out=0.0, dilation_rate=1):
    
    x = Conv2D(filters, kernel_size, activation=None, padding='same', dilation_rate=dilation_rate)(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def bn_act_conv_dense(x, filters, drop_out=0.0):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x


def dense_block(x, elements=3, filters=8, drop_out=0.0):
    
    blocks = []
    blocks.append(x)

    for i in range(elements):
        temp = bn_act_conv_dense(x, filters, drop_out)       
        blocks.append(temp)
#         for j in blocks:
#             print(K.int_shape(j))
        x = Concatenate(axis=-1)(blocks[:])
#         print(K.int_shape(x))
#         x = concatenate(blocks)

    return x

def back_layer(x, filters, drop_out):
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if drop_out > 0:
        x = Dropout(drop_out)(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x
   
def centre_layer(x, filters, drop_out):
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.), bias_regularizer=l2(0.))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.), bias_regularizer=l2(0.))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x
   
def conv2dtranspose(x, filters):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
    return x

def conv2d_o(filters):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=l2(0.),
                  bias_regularizer=l2(0.))


def conv2dtranspose_o(filters):
    return Conv2DTranspose(filters=filters,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same')

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def selective_identity_block(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = selective_layer(x, nb_filter2, compression=0.5,
                         drop_out=0)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    
    x = Conv2D(nb_filter1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=-1)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def hidden_layers(inputs, filters, dropout):
        
        x = Conv2D(filters, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(dropout)(x)
        x = Conv2D(2*filters, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout)(x)
        x = Conv2D(2*filters, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout)(x)
        return x

def dense_branch(inputs, filters, num_grades, dropout):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = hidden_layers(inputs, filters, dropout)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Dense(num_grades)(x)
        x = Activation("softmax")(x)
        return x

def selective_layer(x, filters, compression=0.5, drop_out=0.0):
    x1 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')(x)

    if drop_out > 0:
        x1 = Dropout(drop_out)(x1)

    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters, (3, 3), padding='same')(x)

    if drop_out > 0:
        x2 = Dropout(drop_out)(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = add([x1, x2])

    x3 = GlobalAveragePooling2D()(x3)

    x3 = Dense(int(filters * compression))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x3 = Dense(filters)(x3)

    x3p = Activation('sigmoid')(x3)

    x3m = Lambda(lambda x: 1 - x)(x3p)

    x4 = multiply([x1, x3p])
    x5 = multiply([x2, x3m])

    return add([x4, x5])


def selective_transition_layer(x, filters, drop_out=0.0):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = selective_layer(x, filters, drop_out=drop_out)

    return x


def transition_layer(x, compression, drop_out=0.0):
    n = K.int_shape(x)[-1]

    n = int(n * compression)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n, (1, 1), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x


def attention_layer(d, e, n):
    d1 = Conv2D(n, (1, 1), activation=None, padding='same')(d)

    e1 = Conv2D(n, (1, 1), activation=None, padding='same')(e)

    concat_de = add([d1, e1])

    relu_de = Activation('relu')(concat_de)
    conv_de = Conv2D(1, (1, 1), padding='same')(relu_de)
    sigmoid_de = Activation('sigmoid')(conv_de)

    shape_e = K.int_shape(e)
    upsample_psi = expend_as(sigmoid_de, shape_e[3])

    return multiply([upsample_psi, e])

def unet(filters=16, dropout=0, size=(224, 224, 3), attention_gates=False):
    inp = Input(size)

    c1 = conv_bn_act(inp, filters)
    c1 = conv_bn_act(c1, filters)
    p1 = MaxPooling2D((2, 2))(c1)
    filters = 2 * filters

    c2 = conv_bn_act(p1, filters)
    c2 = conv_bn_act(c2, filters)
    p2 = MaxPooling2D((2, 2))(c2)
    filters = 2 * filters

    c3 = conv_bn_act(p2, filters)
    c3 = conv_bn_act(c3, filters)
    p3 = MaxPooling2D((2, 2))(c3)
    filters = 2 * filters

    c4 = conv_bn_act(p3, filters)
    c4 = conv_bn_act(c4, filters)
    p4 = MaxPooling2D((2, 2))(c4)
    filters = 2 * filters

    cm = conv_bn_act(p4, filters)
    cm = conv_bn_act(cm, filters)

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)

    if attention_gates:

        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)

    else:

        u4 = concatenate([u4, c4], axis=3)

    c5 = conv_bn_act(u4, filters)
    c5 = conv_bn_act(c5, filters)

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)

    if attention_gates:

        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)

    else:

        u3 = concatenate([u3, c3], axis=3)

    c6 = conv_bn_act(u3, filters)
    c6 = conv_bn_act(c6, filters)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)

    if attention_gates:

        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)

    else:

        u2 = concatenate([u2, c2], axis=3)

    c7 = conv_bn_act(u2, filters)
    c7 = conv_bn_act(c7, filters)

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)

    if attention_gates:

        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)

    else:

        u1 = concatenate([u1, c1], axis=3)

    c8 = conv_bn_act(u1, filters)
    c8 = conv_bn_act(c8, filters)

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)

    model = keras.models.Model(inputs = inp, outputs = c9)

    return model
   
def selective_unet(filters=16, drop_out=0, compression=0.5, size=(224, 224, 1),
                   half_net=False, attention_gates=False):
    inp = Input(size)

    c1 = selective_layer(inp, filters, compression=compression,
                         drop_out=drop_out)
    c1 = selective_layer(c1, filters, compression=compression,
                         drop_out=drop_out)
    p1 = MaxPooling2D((2, 2))(c1)
    filters = 2 * filters

    c2 = selective_layer(p1, filters, compression=compression,
                         drop_out=drop_out)
    c2 = selective_layer(c2, filters, compression=compression,
                         drop_out=drop_out)
    p2 = MaxPooling2D((2, 2))(c2)
    filters = 2 * filters

    c3 = selective_layer(p2, filters, compression=compression,
                         drop_out=drop_out)
    c3 = selective_layer(c3, filters, compression=compression,
                         drop_out=drop_out)
    p3 = MaxPooling2D((2, 2))(c3)
    filters = 2 * filters

    c4 = selective_layer(p3, filters, compression=compression,
                         drop_out=drop_out)
    c4 = selective_layer(c4, filters, compression=compression,
                         drop_out=drop_out)
    p4 = MaxPooling2D((2, 2))(c4)
    filters = 2 * filters

    cm = selective_layer(p4, filters, compression=compression,
                         drop_out=drop_out)
    cm = selective_layer(cm, filters, compression=compression,
                         drop_out=drop_out)

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)

    if attention_gates:

        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)

    else:

        u4 = concatenate([u4, c4], axis=3)

    if half_net:

        c5 = conv_bn_act(u4, filters, drop_out=drop_out)
        c5 = conv_bn_act(c5, filters, drop_out=drop_out)

    else:

        c5 = selective_layer(u4, filters, compression=compression,
                             drop_out=drop_out)
        c5 = selective_layer(c5, filters, compression=compression,
                             drop_out=drop_out)

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)

    if attention_gates:

        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)

    else:

        u3 = concatenate([u3, c3], axis=3)

    if half_net:

        c6 = conv_bn_act(u3, filters, drop_out=drop_out)
        c6 = conv_bn_act(c6, filters, drop_out=drop_out)

    else:

        c6 = selective_layer(u3, filters, compression=compression,
                             drop_out=drop_out)
        c6 = selective_layer(c6, filters, compression=compression,
                             drop_out=drop_out)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)

    if attention_gates:

        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)

    else:

        u2 = concatenate([u2, c2], axis=3)

    if half_net:

        c7 = conv_bn_act(u2, filters, drop_out=drop_out)
        c7 = conv_bn_act(c7, filters, drop_out=drop_out)

    else:

        c7 = selective_layer(u2, filters, compression=compression,
                             drop_out=drop_out)
        c7 = selective_layer(c7, filters, compression=compression,
                             drop_out=drop_out)

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)

    if attention_gates:

        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)

    else:

        u1 = concatenate([u1, c1], axis=3)

    if half_net:

        c8 = conv_bn_act(u1, filters, drop_out=drop_out)
        c8 = conv_bn_act(c8, filters, drop_out=drop_out)

    else:

        c8 = selective_layer(u1, filters, compression=compression,
                             drop_out=drop_out)
        c8 = selective_layer(c8, filters, compression=compression,
                             drop_out=drop_out)

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)
    model = keras.models.Model(inputs = inp, outputs = c9)
    
    return model
   
def denseunet(filters=8, blocks=3, layers=3, compression=0.5, drop_out=0,
               size=(224, 224, 1), half_net=True, attention_gates=True):
    
    inp = Input(size)

    x = Conv2D(filters, (3, 3), activation=None, padding='same')(inp)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    names = {}
   
    for i in range(layers):
        
        x = dense_block(x, blocks, filters, drop_out)
#         x = transition_layer(x, compression, drop_out)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)

        name = 'x' + str(i + 1)
        names[name] = x
        
        x = transition_layer(x, compression, drop_out)
        x = MaxPooling2D((2, 2))(x)

        filters = 2 * filters

    x = dense_block(x, blocks, filters, drop_out)
    x = transition_layer(x, compression, drop_out)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

    for i in range(layers):

        filters = filters // 2

        name = 'x' + str(layers - i)

        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        if attention_gates:

            x = Concatenate()([x, attention_layer(x, names[name], 1)])

        else:

            x = Concatenate()([x, names[name]])

        if half_net:

            x = conv_bn_act(x, filters, drop_out)
            x = conv_bn_act(x, filters, drop_out)

        else:

            x = dense_block(x, blocks, filters, drop_out)
            x = transition_layer(x, compression, drop_out)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
    
#     x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(x)

    model = tf.keras.Model(inputs = inp, outputs = x)
 
    return model

def Unetplusplus(size=(224, 224, 1), drop_out=0):
    
    filters = [16,32,64,128,256]
    inp = Input(size)
    X00 = back_layer(inp, filters[0], drop_out)
    PL0 = MaxPooling2D(pool_size=(2, 2))(X00)

    X10 = back_layer(PL0, filters[1], drop_out)
    PL1 = MaxPooling2D(pool_size=(2, 2))(X10)

    X01 = conv2dtranspose(X10, filters[0])
    X01 = concatenate([X00, X01])
    X01 = centre_layer(X01, filters[0], drop_out)

    X20 = back_layer(PL1, filters[2], drop_out)
    PL2 = MaxPooling2D(pool_size=(2, 2))(X20)

    X11 = conv2dtranspose(X20, filters[0])
    X11 = concatenate([X10, X11])
    X11 = centre_layer(X11, filters[0], drop_out)

    X02 = conv2dtranspose(X11, filters[0])
    X02 = concatenate([X00, X01, X02])
    X02 = centre_layer(X02, filters[0], drop_out)

    X30 = back_layer(PL2, filters[3], drop_out)
    PL3 = MaxPooling2D(pool_size=(2, 2))(X30)

    X21 = conv2dtranspose(X30, filters[0])
    X21 = concatenate([X20, X21])
    X21 = centre_layer(X21, filters[0], drop_out)

    X12 = conv2dtranspose(X21, filters[0])
    X12 = concatenate([X10, X11, X12])
    X12 = centre_layer(X12, filters[0], drop_out)

    X03 = conv2dtranspose(X12, filters[0])
    X03 = concatenate([X00, X01, X02, X03])
    X03 = centre_layer(X03, filters[0], drop_out)

    M = centre_layer(PL3, filters[4], drop_out)

    X31 = conv2dtranspose(M, filters[3])
    X31 = concatenate([X31, X30])
    X31 = centre_layer(X31, filters[3], drop_out)

    X22 = conv2dtranspose(X31, filters[2])
    X22 = concatenate([X22, X20, X21])
    X22 = centre_layer(X22, filters[2], drop_out)

    X13 = conv2dtranspose(X22, filters[1])
    X13 = concatenate([X13, X10, X11, X12])
    X13 = centre_layer(X13, filters[1], drop_out)

    X04 = conv2dtranspose(X13, filters[0])
    X04 = concatenate([X04, X00, X01, X02, X03], axis=3)
    X04 = centre_layer(X04, filters[0], drop_out=0.0)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(X04)

    model = tf.keras.Model(inputs=inp, outputs=output)

    return model

    
def GRADE_MODEL(size, dropout=0.1, filters = 16, num_grades = 3):
    inp = Input(size)
    GRADE_branch = dense_branch(inp, filters, num_grades, dropout)
    model = tf.keras.Model(inputs=inp, outputs = GRADE_branch)
    return model

def ResNet101(size, num_grades = 2, filters=16, depth = 23):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 3):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, 23):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def SK_ResNet101(size, num_grades = 3):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [8, 8, 32], strides=(1, 1))
    x = selective_identity_block(x, 3, [16, 16, 32])
    x = selective_identity_block(x, 3, [16, 16, 32])

    x = conv_block(x, 3, [32, 32, 128])
    for i in range(1, 3):
        x = selective_identity_block(x, 3, [32, 32, 128])

    x = conv_block(x, 3, [64, 64, 256])
    for i in range(1, 23):
        x = selective_identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = selective_identity_block(x, 3, [128, 128, 512])
    x = selective_identity_block(x, 3, [128, 128, 512]) #2048

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + "sq1x1")(x)
    x = Activation('relu', name=s_id + "relu" + "sq1x1")(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + "exp1x1")(x)
    left = Activation('relu', name=s_id + "relu" + "exp1x1")(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + "exp3x3")(x)
    right = Activation('relu', name=s_id + "relu" + "exp3x3")(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

def SqueezeNet(size, include_top=True, pooling='avg', classes=2):
    

    inp = Input(size)
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inp)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
        x = Dropout(0.5, name='drop9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = Activation('relu', name='relu_conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)
    x = Flatten()(x)
    x = Dense(classes*50)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    
    model = tf.keras.Model(inputs=inp, outputs = x, name='squeezenet')
    return model

def duo_squeeze(size, include_top=False, pooling='avg', classes=2):
    
    inpA = Input(size)
    inpB = Input(size)

    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inpA)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
        x = Dropout(0.5, name='drop9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = Activation('relu', name='relu_conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)
    y = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inpB)
    y = Activation('relu', name='relu_conv1')(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(y)

    y = fire_module(y, fire_id=2, squeeze=16, expand=64)
    y = fire_module(y, fire_id=3, squeeze=16, expand=64)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(y)

    y = fire_module(y, fire_id=4, squeeze=32, expand=128)
    y = fire_module(y, fire_id=5, squeeze=32, expand=128)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(y)

    y = fire_module(y, fire_id=6, squeeze=48, expand=192)
    y = fire_module(y, fire_id=7, squeeze=48, expand=192)
    y = fire_module(y, fire_id=8, squeeze=64, expand=256)
    y = fire_module(y, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
        y = Dropout(0.5, name='drop9')(y)

        y = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(y)
        y = Activation('relu', name='relu_conv10')(y)
        y = GlobalAveragePooling2D()(y)
        y = Activation('softmax', name='loss')(y)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)

    combined = concatenate([x, y])

    z = Flatten()(combined)
    z = Dense(classes*50)(z)
    z = Activation("relu")(z)
    z = BatchNormalization()(z)

    z = Dense(classes)(z)
    z = Activation("softmax")(z)
    
    model = tf.keras.Model(inputs=inp, outputs = z, name='duo_squeezenet')

def SIMPLE_CON(size, num_grades = 2, filters=8, compression = 0.5, drop_out = 0):
    inp = Input(size)

    x = selective_layer(inp, filters, compression=compression, drop_out=drop_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(64, (5, 5), strides=(2, 2),name="layer1")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = AveragePooling2D((5, 5),name="last_layer")(x)

    x = Flatten()(x)
    

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model


def CON_1(size, num_grades = 2, filters=8, compression = 0.5, drop_out = 0):
    inp = Input(size)

    x = selective_layer(inp, filters, compression=compression,
                         drop_out=drop_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2),name="layer1")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = selective_layer(inp, filters, compression=compression,
                         drop_out=drop_out)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = AveragePooling2D((5, 5),name="last_layer")(x)

    x = Flatten()(x)
    

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model


def CON_2(size, num_grades = 2, filters=8, compression = 0.5, drop_out = 0):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    p1 = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(p1, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x,p1])
    p2 = MaxPooling2D((2, 2))(x)
    
    x = conv_block(p2, 3, [4*filters, 4*filters, 16*filters], strides=(1, 1))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x,p2])
    p3 = MaxPooling2D((2, 2))(x)

    x = conv_block(p3, 3, [16*filters, 16*filters, 64*filters], strides=(1, 1))

    for i in range(0, 5):
        x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    

    x = concatenate([x,p3])
    x = AveragePooling2D((7, 7),name="outlayer")(x)
    x = Flatten()(x)
    

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model


def identity_block2(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block2(input_tensor, kernel_size, filters, strides=(2, 2)):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    
    x = Conv2D(nb_filter1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=-1)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet2(depth, size, num_grades = 2, filters=8):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 3):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, depth):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def attention56(size, num_grades = 2, filters=8):
    
    
    inp = Input(size)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(inp)
    p1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p1, 3, [8*filters, 8*filters, 32*filters], strides=(1, 1))
    x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = concatenate([x, attention_layer(x, p1, 1)], axis=3)
    p2 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p2, 3, [16*filters, 16*filters, 64*filters], strides=(1, 1))
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = concatenate([x, attention_layer(x, p2, 1)], axis=3)
    p3 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p3, 3, [32*filters, 32*filters, 128*filters], strides=(1, 1))
    x = identity_block(x, 3, [32*filters, 32*filters, 128*filters])
    x = concatenate([x, attention_layer(x, p3, 1)], axis=3)
    p4 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p4, 3, [64*filters, 64*filters, 256*filters], strides=(1, 1))
    x = identity_block(x, 3, [64*filters, 64*filters, 256*filters])
    x = AveragePooling2D((2, 2),name="outlayer")(x)
    x = Flatten()(x)
    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   
    
    
  
def VGV(size, num_grades = 2, filters=16):
    
    inp = Input(size)
    x = Conv2D(8*filters, (3, 3), padding="same", activation='relu')(inp)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(16*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(16*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(32*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(32*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(32*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(64*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(64*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(128*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(128*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(128*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4069, activation="relu")(x)
    x = Dense(4069, activation="relu")(x)
    x = Dense(num_grades, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs = x)
    return model 

def ResNet50(size, num_grades = 2, filters=64):

    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [filters, filters, 4*filters])
    x = identity_block(x, 3, [filters, filters, 4*filters])

    x = conv_block(x, 3, [2*filters, 2*filters, 8*filters])
    for i in range(0, 3):
        x = identity_block(x, 3, [2*filters, 2*filters, 8*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(0, 5):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = identity_block(x, 3, [8*filters, 8*filters, 32*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def GAO_NET(size, num_grades = 2, filters=8):
    inp = Input(size)
    x = Conv2D(filters, (7, 7), strides=(2, 2))(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(3, 3), padding='same')(x)
    x = Conv2D(96*filters, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(256*filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(384*filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(384*filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dense(2048)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

    
    
    
def DilatedSpatialPyramidPooling(dspp_input):
    
    dims = dspp_input.shape
    print(dims)
    
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    
    x = conv_bn_act(x, filters=256, kernel_size=1)
    
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = conv_bn_act(dspp_input, filters=256, kernel_size=1, dilation_rate=1)
    out_6 = conv_bn_act(dspp_input, filters=256, kernel_size=3, dilation_rate=4)
    out_12 = conv_bn_act(dspp_input, filters=256, kernel_size=3, dilation_rate=8)
    out_18 = conv_bn_act(dspp_input, filters=256, kernel_size=3, dilation_rate=12)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    
    output = conv_bn_act(x, filters=256, kernel_size=1)
    
    return output

def DeeplabV3(image_size=512):
    
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    #resnet = pretrained_ResNet101(weights=None, include_top=False, input_tensor=model_input)
    resnet = tf.keras.applications.Xception(weights=None, include_top=False, input_tensor=model_input)
    
    #resnet.trainable = False
    resnet.summary()
    x = resnet.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear")(x)
    
    input_b = resnet.get_layer("conv2_block3_2_relu").output
    input_b = conv_bn_act(input_b, filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = conv_bn_act(x, filters=256)
    x = conv_bn_act(x, filters=256)
    x = UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)
    
    model_output = Conv2D(1, kernel_size=1, padding="same", activation='sigmoid')(x)
    
    return Model(inputs=model_input, outputs=model_output) 

def conv2d_same(x, filters, stride=1, kernel_size=3, rate=1):
    if stride == 1:
        return Conv2D(filters,(kernel_size, kernel_size),strides=(stride, stride),padding='same', use_bias=False,dilation_rate=(rate, rate))(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,(kernel_size, kernel_size),strides=(stride, stride),padding='valid', use_bias=False,dilation_rate=(rate, rate))(x)

def Depth_Sep_Conv(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    return x


def Xcept_block(inputs, depth_list, skip_connection_type, stride,rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = Depth_Sep_Conv(residual,depth_list[i],stride=stride if i == 2 else 1,rate=rate,depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = conv2d_same(inputs, depth_list[-1],kernel_size=1,stride=stride)
        shortcut = BatchNormalization()(shortcut)
        outputs = Add()([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = Add()([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

    return x
    
def Xception_deeplab(filters=32, dropout=0, size=(224, 224, 3), attention_gates=False):
    inp = Input(size)
    #x = ZeroPadding2D((3, 3), name='Intial padding')(inp)
    x = Conv2D(filters, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv2d_same(x, 2*filters, stride=1, kernel_size=3, rate=1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Xcept_block(x, [4*filters, 4*filters, 4*filters], skip_connection_type='conv', stride=2, depth_activation=False)
    x = Xcept_block(x, [8*filters, 8*filters, 8*filters],skip_connection_type='conv', stride=2, depth_activation=False)
    x, skip1 = Xcept_block(x, [728, 728, 728], skip_connection_type='conv', stride=2, depth_activation=False, return_skip=True)
    for i in range(0, 16):
        x = Xcept_block(x, [728, 728, 728], skip_connection_type='sum', stride=1, rate=1, depth_activation=False)        
    x = Xcept_block(x, [728, 1024, 1024], skip_connection_type='conv', stride=2, rate=1,depth_activation=False)
    x = Xcept_block(x, [1536, 1536, 2048], skip_connection_type='conv', stride=1, rate=2,depth_activation=False)
    x = DilatedSpatialPyramidPooling(x)
    image_size = 224
    input_a = UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear")(x)
    input_b = conv_bn_act(skip1, filters=256, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = conv_bn_act(x, filters=256)
    x = conv_bn_act(x, filters=256)
    x = UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)
    model_output = Conv2D(1, kernel_size=1, padding="same", activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs = model_output)
    return model
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
