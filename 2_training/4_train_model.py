yellow_path = "//wsl$/Ubuntu-20.04/home/afer/yellow/"
import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/');
import ml_utils
from patch_extraction import blocks_to_raster

import pandas as pd
import numpy as np
import os
import math

from sklearn.model_selection import train_test_split
import segmentation_models as sm

import matplotlib.pyplot as plt

# Tensorflow
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
import tensorflow_addons as tfa

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

arrays_dir = "R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/patches_arrays/"

X1 = np.load(arrays_dir + "dtm_aeroe_64.npy")
X2 = np.load(arrays_dir + "sobel_64.npy")
X3 = np.load(arrays_dir + "hat_aeroe_64.npy")
y = np.load(arrays_dir + "walls_10km_611_57_binpatches_64.npy")

X = np.stack((X1, X2, X3), axis=3)

print(X.shape)
print(y.shape)

X1_abs = np.load(arrays_dir + "dtm_aeroe_64.npy")
X2_abs = np.load(arrays_dir + "sobel_64.npy")
X3_abs = np.load(arrays_dir + "hat_aeroe_64.npy")
y_abs = np.load(arrays_dir + "absence_10km_611_57_binpatches_64.npy")

X_abs = np.stack((X1_abs, X2_abs, X3_abs), axis=3)
X_abs = X_abs[..., 0]

print(X_abs.shape)
print(y_abs.shape)

X = np.concatenate([X, X_abs])
y = np.concatenate([y, y_abs])

shuffle_mask = np.random.permutation(len(y))
X = X[shuffle_mask]
y = y[shuffle_mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X train: ", X_train.shape, "y train: ", y_train.shape, "y_train datatype: ", y_train.dtype)
print("X test: ", X_test.shape, "y test: ", y_test.shape, "y_test datatype: ", y_test.dtype)

X_train = ml_utils.add_rotations(X_train)
y_train = ml_utils.add_rotations(y_train)

if y_train.dtype != "float32":
    y_train = y_train.astype(np.float32)
    print(y_train.dtype)

if y_test.dtype != "float32":
    y_test = y_test.astype(np.float32)
    print(y_test.dtype)

shape = (X_train.shape[1:])
print(shape)


def define_model(shape, name, activation='relu', sizes=[32, 64, 96, 128], double=True, pool="max", 
                 padding=["same", "valid"], drop=0.2):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(sizes[0], kernel_size=3,
        padding=padding[0],
        activation=activation,
        #kernel_initializer=kernel_initializer
    )(model_input)
    if double:
        modelskip1 = Conv2D(sizes[0], kernel_size=3,
            padding=padding[0],
            activation=activation,
            #kernel_initializer=kernel_initializer
        )(model)
        model = BatchNormalization()(modelskip1)
        model = Dropout(drop)(model)
    if pool == 'max':
          model = MaxPool2D(padding=padding[0])(model)
    if pool == 'average':
          model = AveragePooling2D(padding=padding[0])(model)
    
    model = Conv2D(sizes[1], kernel_size=3,
        padding=padding[0],
        activation=activation,
        #kernel_initializer=kernel_initializer
    )(model)
    if double:
        modelskip2 = Conv2D(sizes[1], kernel_size=3,
            padding=padding[0],
            activation=activation,
            #kernel_initializer=kernel_initializer
        )(model)
        model = BatchNormalization()(modelskip2)
        model = Dropout(drop)(model)
    if pool == 'max':
        model = MaxPool2D(padding=padding[0])(model)
    if pool == 'average':
        model = AveragePooling2D(padding=padding[0])(model)
    
    model = Conv2D(sizes[2], kernel_size=3,
        padding=padding[0],
        activation=activation,
        #kernel_initializer=kernel_initializer
    )(model)
    if double:    
        model = Conv2D(sizes[2], kernel_size=3,
            padding=padding[0],
            activation=activation,
            #kernel_initializer=kernel_initializer
        )(model)
        model = BatchNormalization()(model)
        model = Dropout(drop)(model)

    model = Conv2DTranspose(sizes[1], kernel_size=3,
        strides=(2,2),
        padding=padding[0],
        activation=activation,
        #kernel_initializer=kernel_initializer
    )(model)
#     model = BatchNormalization()(model)
#     model = Dropout(drop)(model)

    model = Concatenate()([modelskip2, model])
    model = Conv2DTranspose(sizes[0], kernel_size=3,
        strides=(2,2),
        padding=padding[0],
        activation=activation,
        #kernel_initializer=kernel_initializer
    )(model)
#     model = BatchNormalization()(model)
#     model = Dropout(drop)(model)
    model = Concatenate()([modelskip1, model])

    output = Conv2D(1, kernel_size=3, padding=padding[0], activation='sigmoid')(model)
    return Model(inputs=[model_input], outputs=output)

model = define_model(shape, name="vanilla binary")
model.summary()



lr = 0.001

model.compile(
    optimizer=Adam(learning_rate=lr),
    loss= sm.losses.binary_focal_jaccard_loss,
    metrics=[sm.metrics.iou_score, 'accuracy'],
)

def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

results = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=32,
    callbacks=[
        LearningRateScheduler(step_decay),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            #min_delta=10,
            restore_best_weights=True,
        ),
    ]
)

accuracy_test = model.evaluate(X_test, y_test)

out_directory = 'R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/predictions/10km_611_57'
unique_filename = out_directory + 'm1_bin_jl_64_611_57' + '.h5'
model.save(unique_filename)
