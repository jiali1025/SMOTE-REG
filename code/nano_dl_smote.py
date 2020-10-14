# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from sklearn.model_selection import KFold
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC, RandomOverSampler
from sklearn.preprocessing import StandardScaler

ss = StandardScaler( )

dataLocation = '../data/'
resultSaveLocation = '../results/'

data2 = pd.read_csv(dataLocation + 'dataset-NR-aspectratio.csv')

data2VariableNames = data2.columns.tolist()
feaColums = data2VariableNames[1:4]
labels = data2VariableNames[4:]

data2Del = data2.drop_duplicates(subset=feaColums,keep='first',inplace=False)

ind_list = [i for i in range(data2Del.shape[0])]

ind_set = list(itertools.combinations(ind_list,2))

model_smote = SMOTE(k_neighbors=1,random_state=0)

data_smote_all = []
ind_smote_all = []
ind_smote = np.zeros(data2Del.shape[0]-2)
ind_smote[:2] = 1
for item in ind_set:
    ind_ = list(item)
    y_smote = np.zeros(data2Del.shape[0])
    y_smote[ind_] = 1
    data_smote_resampled , y_smote_resampled = model_smote.fit_resample(data2Del , y_smote)
    ind = np.where(y_smote_resampled == 1)
    data_ = data_smote_resampled[ind]
    data_smote_all.append(data_)
    ind_smote_all.append(ind_smote)

data_smote_all = np.array(data_smote_all)
data_smote_all = np.reshape(data_smote_all , [-1 , 7])

ind_smote_all = np.array(ind_smote_all)
ind_smote_all = np.reshape(ind_smote_all , [-1 , 1])

Dataover = np.concatenate((data_smote_all , ind_smote_all) , axis=1)
Dataover = pd.DataFrame(Dataover)
Dataover.columns = data2VariableNames + ['ind_smote']

Dataover = Dataover.drop(columns=data2VariableNames[0])

Datasave = Dataover.drop_duplicates(subset=feaColums,keep='first' , inplace=False)
Datasave = Datasave.sample(frac=1)

Dataover.to_csv(resultSaveLocation + 'nanorod_smote.csv')
Datasave.to_csv(resultSaveLocation + 'nanorod_smote_dupshu.csv')

a


