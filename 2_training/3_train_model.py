#%%
import os
# import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import Model, Input
import random
from sklearn.preprocessing import MinMaxScaler
#from keras.utils import to_categorical
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model
from tensorflow.python.client import device_lib


from tensorflow.keras.utils import to_categorical

from sklearn.model_selection  import train_test_split

import pickle

import glob

#%%

#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


os.environ["CUDA_VISIBLE_DEVICES"]="0"

sm.set_framework('tf.keras')

sm.framework()

tf.config.list_physical_devices('GPU')

device_lib.list_local_devices()


#%%
##data load
x_train = []
y_train = []

files = glob.glob(r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\npz_tiles_data\*.npz')

for fn in files:
    # print(fn)
    with np.load(fn) as data:
        # with np.load(fr'{trainingdatadir}/data_test.npz') as data:
        x = data['x']
        y = data['y']

        x_train.append(x)
        y_train.append(y)

#%%
##no data load
files = glob.glob(r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\npz_tiles_nodata\*.npz')

for fn in files:
    # print(fn)
    with np.load(fn) as data:
        # with np.load(fr'{trainingdatadir}/data_test.npz') as data:
        x = data['x']
        y = data['y']

        x_train.append(x)
        y_train.append(y)
#%%
##concatenate and shuffle data
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

#%%
# shuffle_mask = np.random.permutation(len(y))
# x_train = x_train[shuffle_mask]
# y_train = y_train[shuffle_mask]

#Use this to preprocess input for transfer learning
# BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)
# x = preprocess_input(x)

#x = x/255


#%%
##divide into train and val
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=42)


print(x_train.dtype, x_val.dtype, y_train.dtype, y_val.dtype)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

#%%

shape = (x_train.shape[1:])
print(shape)

#%%

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


#%%
##compile model
lr = 0.001

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss= sm.losses.binary_focal_jaccard_loss,
    metrics=[sm.metrics.iou_score, 'accuracy'],
)


def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

model.summary()

#%%
# tf.keras.backend.clear_session()

# model = sm.Unet(BACKBONE, encoder_weights='imagenet', 
#                 input_shape=(512, 512, 3),
#                 classes=5, activation='softmax')


# # co = {
# #     'focal_loss_plus_jaccard_loss': sm.losses.categorical_focal_jaccard_loss,
# #     'iou_score': sm.metrics.iou_score
# # }


# # model.build((None,512,512,3))

# adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
# lossfn = sm.losses.JaccardLoss(class_indexes=[0,1,2,3])+sm.losses.CategoricalFocalLoss(class_indexes=[0,1,2,3])

# # lossfn = sm.losses.categorical_focal_jaccard_loss

# model.compile(optimizer=adam, loss=lossfn, metrics=[sm.metrics.iou_score, 'accuracy'])

# model.summary()

#%%
##add rotations to x_train and y_train
def add_rotations(X, k=4, axes=(1, 2)):
    if k == 1:
        return X
    elif k == 2:
        return np.concatenate([X, np.rot90(X, k=2, axes=axes),])
    elif k == 3:
        return np.concatenate(
            [X, np.rot90(X, k=1, axes=axes), np.rot90(X, k=2, axes=axes),]
        )
    else:
        return np.concatenate(
            [
                X,
                np.rot90(X, k=1, axes=axes),
                np.rot90(X, k=2, axes=axes),
                np.rot90(X, k=3, axes=axes),
            ]
        )

x_train = add_rotations(x_train)
y_train = add_rotations(y_train)

print(len(x_train))

# idg = tf.keras.preprocessing.image.ImageDataGenerator(
#     horizontal_flip=True,
#     vertical_flip=True,
#     rotation_range = 45,
#     # preprocessing_function=add_rotations(x),
#     #height_shift_range=100,
#     #preprocessing_function=rc
# )


# seed = 100
# bz = 6

# genx = idg.flow(
#     x_train,seed=seed,batch_size=bz,shuffle=True,
#     #save_to_dir=yield_dir
# )
# geny = idg.flow(
#     y_train,seed=seed,batch_size=bz,shuffle=True,
#     #save_to_dir=yield_dir
# )
# gen = zip(genx,geny)





#%%

# idg_val= tf.keras.preprocessing.image.ImageDataGenerator(
#     # horizontal_flip=False,
#     # vertical_flip=False
#     #preprocessing_function=preprocess_input,
#     #height_shift_range=100,
#     #preprocessing_function=rc
# )

# seed = 100
# bz = 6

# genx_val = idg_val.flow(
#     x_val,seed=seed,batch_size=bz,shuffle=False,
#     #save_to_dir=yield_dir
# )
# geny_val = idg_val.flow(
#     y_val,seed=seed,batch_size=bz,shuffle=False,
#     #save_to_dir=yield_dir
# )
# gen_val = zip(genx_val,geny_val)


#def gen_val(xv, yv, bz):


#%%
# def val_generator(xvs, yvs):
#     while True:
#         for v in zip(xvs, yvs):
#             yield v

# def createVal(xv,yv,bz):

#     xvs = np.array_split(xv, np.ceil(len(xv)/bz))
#     yvs = np.array_split(yv, np.ceil(len(yv)/bz))

#     return len(xvs), val_generator(xvs, yvs)



# steps_val, gen_val = createVal(x_val, y_val, bz)




#%%
##define callbacks
logdir = 'V:/2022-03-31_Stendiger_EZRA/code/logs/'

callbacks= [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=logdir+'/model.{epoch:02d}-{val_loss:.3f}-{val_iou_score:.3f}-{val_accuracy:.3f}.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        # min_delta=0,
        patience=10,
        verbose=0,
        # mode='max',
        # baseline=None,
        restore_best_weights=True
),
    tf.keras.callbacks.LearningRateScheduler(step_decay)
    ]


#%%
##train model

history=model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    # validation_data=gen_val,
    # validation_steps=steps_val,
    # validation_batch_size=None,
    # steps_per_epoch=200,
    initial_epoch=0,
    epochs=100,
    verbose=1,
    callbacks=[callbacks],
)

#%%
# plt.plot(history.history['val_iou_score'])

# plt.plot(history.history['iou_score'])

#%%

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();

#%%

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(history.history['iou_score'], label='iou_score')
plt.plot(history.history['val_iou_score'], label='val_iou_score')
plt.plot( np.argmin(history.history['val_iou_score']), np.min(history.history['val_iou_score']), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel('iou_score')
plt.legend();

#%%
with open(f'{logdir}/history.pkl', 'wb') as pickle_file:
    pickle.dump(history.history, pickle_file)

# pickle.dump(history, f'{logdir}/history.pkl')
#%%

print('done training!')
