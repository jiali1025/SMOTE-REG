# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import json
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

dataLocation = '../results/'
resultSaveLocation = '../RESULTS/DeepLearning_val'
if not os.path.exists(resultSaveLocation):
    os.makedirs(resultSaveLocation)

data2 = pd.read_csv(dataLocation + 'nanorod_smote_dupshu.csv')

data2VariableNames = data2.columns.tolist()
feaColums = data2VariableNames[1:4]
indSmote = data2VariableNames[-1]

label_one = data2VariableNames[-2]

lr = 0.001
dense_size = 64
validation_split = 0.2
resultSaveLocation = resultSaveLocation + '/' + str(lr) + '/' + str(dense_size) + '/' + label_one + '/'
if not os.path.exists(resultSaveLocation):
    os.makedirs(resultSaveLocation)

data2Del = data2.drop_duplicates(subset=feaColums,keep='first',inplace=False)

ind_smote = data2Del[indSmote]
X = np.array(data2Del[feaColums])
Y = np.array(data2Del[label_one])
Y = np.reshape(Y,[-1,1])

nfold = 10
kf = KFold(n_splits=nfold, shuffle = False, random_state=1)
ss = StandardScaler( )

inputsize = X.shape[1]

def build_model ():
    model = keras.Sequential([
        layers.Dense(dense_size, activation=tf.nn.relu, input_shape=[inputsize]),
        layers.Dense(dense_size, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(lr)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model( )

model.summary( )

nfold_train_data = []
nfold_train_Y = []
nfold_test_data = []
nfold_test_Y = []
nfold_test_ind = []

for train, test in kf.split(X):
    train_X = X[train]
    train_Y = Y[train]
    test_X = X[test]
    test_Y = Y[test]
    # train_ind = ind_smote[train]
    test_ind = ind_smote[test]

    train_X = ss.fit_transform(train_X)
    test_X = ss.transform(test_X)

    nfold_train_data.append(train_X)
    nfold_train_Y.append(train_Y)
    nfold_test_data.append(test_X)
    nfold_test_Y.append(test_Y)
    nfold_test_ind.append(test_ind)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end (self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

Ytest =[]
Yhat = []

import matplotlib.pyplot as plt
def plot_history (history,ncv):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure( )
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend( )
    plt.ylim([0, 5])
    plt.savefig(resultSaveLocation + str(EPOCHS) +'_'+ str(ncv+1) +'_' + str(validation_split) + '_mae.pdf')

    plt.figure( )
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend( )
    plt.ylim([0, 20])
    plt.savefig(resultSaveLocation + str(EPOCHS) +'_'+ str(i+1) +'_'+ str(validation_split) + '_mse.pdf')

loss_cv = []
mae_cv = []
mse_cv = []
ind_cv = []
r2_cv = []

for i in range(nfold):
    X = nfold_train_data[i]
    Y = nfold_train_Y[i]
    Xt = nfold_test_data[i]
    Yt = nfold_test_Y[i]
    ind_test = nfold_test_ind[i]

    model = build_model( )
    history = model.fit(
        X, Y,
        epochs=EPOCHS, validation_split=validation_split, verbose=0,
        callbacks=[PrintDot( )])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plot_history(history,i)

    # The patience parameter is the amount of epochs to check for improvement
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss' , patience=10)

    loss, mae, mse = model.evaluate(Xt, Yt, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
    loss_cv.append(loss)
    mae_cv.append(mae)
    mse_cv.append(mse)

    # test_predictions = model.predict(nfold_test_data[i]).flatten()
    Yhati = model.predict(Xt)

    model.save(resultSaveLocation+label_one+ str(EPOCHS) + '_' + str(validation_split)  + '_CV' + str(i) + '_model.h5')

    r2 = r2_score(Yhati , Yt)
    r2_cv.append(r2)

    Ytest.append(Yt.flatten())
    Yhat.append(Yhati.flatten())
    ind_cv.append(np.array(ind_test))

Ytestj = [list(str(iitem) for iitem in item) for item in Ytest]
Yhatj = [list(str(iitem) for iitem in item) for item in Yhat]
indj = [list(str(iitem) for iitem in item) for item in ind_cv]
results = {'Ytest':Ytestj,'Yhat':Yhatj,'ind_smote':indj}
with open(resultSaveLocation + label_one + str(EPOCHS) + '_' + str(validation_split) +  '_results.json', 'w') as f:
    json.dump(results, f)

loss_cv.append(str(np.mean(loss_cv)) + '+-' + str(np.std(loss_cv)))
mae_cv.append(str(np.mean(mae_cv)) + '+-' + str(np.std(mae_cv)))
mse_cv.append(str(np.mean(mse_cv)) + '+-' + str(np.std(mse_cv)))
r2_cv.append(str(np.mean(r2_cv)) + '+-' + str(np.std(r2_cv)))
measures = np.concatenate((np.reshape(loss_cv,(-1,1)),np.reshape(mae_cv,(-1,1)),np.reshape(mse_cv,(-1,1)),np.reshape(r2_cv,(-1,1))),axis=1)
measures = pd.DataFrame(measures)
measures.columns = ['loss','mae','mse','r2']
measures.to_csv(resultSaveLocation+label_one+ str(EPOCHS) + '_' + str(validation_split)  +'_measures.csv')

Ytests = Ytest[0]
Yhats = Yhat[0]
inds = ind_cv[0]
for i in range(nfold-1):
    Ytests = np.concatenate((Ytests,Ytest[i+1]),axis=0)
    Yhats = np.concatenate((Yhats,Yhat[i+1]),axis=0)
    inds = np.concatenate((inds,ind_cv[i+1]),axis=0)

resultsY = np.concatenate((np.reshape(Ytests,(-1,1)),np.reshape(Yhats,(-1,1)),np.reshape(inds,(-1,1))),axis=1)
resultsY = pd.DataFrame(resultsY)
resultsY.columns = ['Ytest','Yhat','ind_smote']
resultsY.to_csv(resultSaveLocation+ str(EPOCHS) + '_'  + str(validation_split) +'_resultsY.csv')

import utils.myplots as myplots
myplots.scatterresults(Ytests, Yhats,'Deep Learning',str(EPOCHS) +'_' + str(validation_split) + '_' + label_one, resultSaveLocation)


