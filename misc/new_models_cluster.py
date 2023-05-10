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


def conv_bn_act(x, filters, drop_out=0.0):
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

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

def unet(filters=16, dropout=0, size=(224, 224, 1), attention_gates=False):
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
def ResNet101(size, num_grades = 1, filters=32):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 3):
        x = identity_block(x, 3, [4*filters, 14*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, 23):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    #x = Flatten()(x)
    #x = Dense(128)(x)
    #x = Activation("relu")(x)
    #x = BatchNormalization()(x)

    #x = Dense(num_grades)(x)
    #x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def SK_ResNet101(size, num_grades = 2):
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

