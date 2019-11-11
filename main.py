import os
from __future__ import print_function
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn import preprocessing
from tensorflow.compat.v1.keras.utils import normalize
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
import pandas as pd
import time
import pickle


def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

nb_epochs = 100
batch_size = 128
fname  = 'Earthquakes'

x_train, y_train = readucr(fname+'_TRAIN')
x_test, y_test = readucr(fname+'_TEST')

#Add generated data
with open('generated-P2.pkl', 'rb') as g1: 
        sig_gen = pickle.load(g1)
x_gen = sig_gen[:206,1:]
y_gen = sig_gen[:206,0]
x_train = np.concatenate((x_train, x_gen), axis=0)
y_train = np.concatenate((y_train, y_gen), axis=0)

nb_classes = len(np.unique(y_test))
y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)

x_test = (x_test - x_train_mean)/(x_train_std)
x_train = x_train.reshape(x_train.shape + (1,1,))
x_test = x_test.reshape(x_test.shape + (1,1,))

x = keras.layers.Input(x_train.shape[1:])
# drop_out = Dropout(0.5)(x)
conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
conv1 = keras.layers.normalization.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)

# drop_out = Dropout(0.6)(conv1)
conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
conv2 = keras.layers.normalization.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

# drop_out = Dropout(0.7)(conv2)
conv3 = keras.layers.Conv2D(128, 3, 1, border_mode='same')(conv2)
conv3 = keras.layers.normalization.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)    
out = keras.layers.Dense(nb_classes, activation='softmax')(full)
                    
test = []
train = []
val = []
for itr in range(10):
    
    print("** Iteration " +str(itr) +" *")
    model = Model(input=x, output=out)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    path_checkpoint = 'checkpoint-' +str(itr) +'-pochs=' +str(nb_epochs) +'-batch=' +str(batch_size) +'.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_weights_only=True,
                                    save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,min_lr=0,patience=10,verbose=1)
    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
            verbose=0, validation_data=(x_test, Y_test),
            callbacks=[callback_checkpoint, reduce_lr])
    log = pd.DataFrame(hist.history)

################### TEST DATA
    tr = 0
    fl = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    y_predicted = model.predict(x_test)

    for i in range(len(Y_test)):
        if  np.argmax(Y_test[i]) == np.argmax(y_predicted[i]):
            tr += 1
            if np.argmax(Y_test[i]) == 0:
                tn += 1
            elif np.argmax(Y_test[i]) == 1:
                tp += 1

        else:
            fl += 1
            if np.argmax(Y_test[i]) == 0:
                fn += 1
            elif np.argmax(Y_test[i]) == 1:
                fp += 1
    test.append(float(tr)/(tr+fl))
    train.append(log.loc[log['loss'].idxmin]['acc'])
    val.append(log.loc[log['loss'].idxmin]['val_acc'])
    print("** TEST DATA ==> True = " +str(tr) +"\tFalse = " +str(fl) +"\t==>\t" +str(float(tr)/(tr+fl)))
    print("TP: " +str(tp) +"\tTN: " +str(tn) +"\tFP: " +str(fp) +"\tFN: " +str(fn) +"\n")

for i in range(len(test)):
    print(str(train[i]) +"\t" +str(val[i]) +"\t" +str(test[i]))

print("The mean in train set is : " +str(np.asarray(train).mean()))
print("The mean in validation set is : " +str(np.asarray(val).mean()))
print("The mean in test set is : " +str(np.asarray(test).mean()))
