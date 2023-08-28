import glob
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from natsort import natsort

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

#
# np.random.seed(0)
# tf.set_random_seed(0)
tf.random.set_seed(0)

train_list = glob.glob('dataset/data_split/train/*')
val_list = glob.glob('dataset/data_split/val/*')
test_list = glob.glob('dataset/data_split/test/*')

train_list = natsort.natsorted(train_list)
val_list = natsort.natsorted(val_list)
test_list = natsort.natsorted(test_list)


def make_dataset(data, label, window_size=100):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(label.iloc[i + window_size]))
    return np.array(feature_list), np.array(label_list)


x_train_arr, x_val_arr, x_test_arr = [], [], []
y_train_arr, y_val_arr, y_test_arr = [], [], []

for i in range(len(train_list)):
    train_pd = pd.read_csv(train_list[i], engine='python')
    val_pd = pd.read_csv(val_list[i], engine='python')
    test_pd = pd.read_csv(test_list[i], engine='python')

    train_pd.columns = ['index', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'label']
    train_pd = train_pd.drop(['index'], axis=1)

    val_pd.columns = ['index', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'label']
    val_pd = val_pd.drop(['index'], axis=1)

    test_pd.columns = ['index', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'label']
    test_pd = test_pd.drop(['index'], axis=1)

    # print(train.head())

    print(train_pd.shape, val_pd.shape, test_pd.shape)  # (45149, 9) (15149, 9) (15149, 9)

    feature_cols = ['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8']
    label_cols = ['label']

    '''feature/label split'''
    train_feature = train_pd[feature_cols]
    train_label = train_pd[label_cols]

    val_feature = val_pd[feature_cols]
    val_label = val_pd[label_cols]

    test_feature = test_pd[feature_cols]
    test_label = test_pd[label_cols]

    # # make dataset 함수에 넣기 위해 data frame 형식으로 변환
    # train_feature = pd.DataFrame(train_feature)
    # train_feature.columns = feature_cols
    #
    # val_feature = pd.DataFrame(val_feature)
    # val_feature.columns = feature_cols
    #
    # test_feature = pd.DataFrame(test_feature)
    # test_feature.columns = feature_cols

    '''make dataset'''
    x_train, y_train = make_dataset(train_feature, train_label)
    x_valid, y_valid = make_dataset(val_feature, val_label)
    x_test, y_test = make_dataset(test_feature, test_label)

    x_train_arr.append(x_train)
    y_train_arr.append(y_train)
    x_val_arr.append(x_valid)
    y_val_arr.append(y_valid)
    x_test_arr.append(x_test)
    y_test_arr.append(y_test)

    # print(x_train.shape, y_train.shape)   # (45049, 100, 8) (45049, 1)
    # print(x_valid.shape, y_valid.shape)   # (15049, 100, 8) (15049, 1)
    # print(x_test.shape, y_test.shape)     # (15049, 100, 8) (15049, 1)

print(np.array(x_train_arr).shape, np.array(y_train_arr).shape)  # (5,) (5,)
print(np.array(x_val_arr).shape, np.array(y_val_arr).shape)      # (5, 12149, 20, 8) (5, 12149, 1)
print(np.array(x_test_arr).shape, np.array(y_test_arr).shape)    # (5, 12149, 20, 8) (5, 12149, 1)

# print(y_train_arr[0, :5])
# print(y_val_arr[0, :5])
# print(y_test_arr[0, :5])

x_train_full = np.concatenate(x_train_arr)
x_val_full = np.concatenate(x_val_arr)
x_test_full = np.concatenate(x_test_arr)
y_train_full = np.concatenate(y_train_arr)
y_val_full = np.concatenate(y_val_arr)
y_test_full = np.concatenate(y_test_arr)

print(x_train_full.shape, y_train_full.shape)   # (180741, 20, 8) (180741, 1)
print(x_val_full.shape, y_val_full.shape)       # (60745, 20, 8) (60745, 1)
print(x_test_full.shape, y_test_full.shape)     # (60745, 20, 8) (60745, 1)

print(x_train_full[0:3], y_train_full[:3])
print(x_val_full[0:3], y_val_full[:3])
print(x_test_full[0:3], y_test_full[:3])
'''scaling'''
scaler = MinMaxScaler()
for i in range(len(x_train_full)):
    scaler.partial_fit(x_train_full[i])

x_train_scaled, x_val_scaled, x_test_scaled = [], [], []
for i in range(len(x_train_full)):
    x_train_scaled.append(scaler.transform(x_train_full[i]))
for i in range(len(x_val_full)):
    x_val_scaled.append(scaler.transform(x_val_full[i]))
    x_test_scaled.append(scaler.transform(x_test_full[i]))

print(np.array(x_train_scaled).shape, np.array(x_val_scaled).shape, np.array(x_test_scaled).shape)
# (225241, 100, 8) (75245, 100, 8) (75245, 100, 8)
print(x_train_scaled[0])

# 객체를 pickled binary file 형태로 저장한다
file_name = 'model/rssi_ae_cnn_mms_full_' + str(ws) + '.pkl'
joblib.dump(scaler, file_name)

'''numpy save'''
np.savez('model/rssi_ae_cnn_data_full_' + str(ws) + '.npz',
         x_train=x_train_scaled,
         x_valid=x_val_scaled,
         x_test=x_test_scaled,
         y_train=y_train_full,
         y_valid=y_val_full,
         y_test=y_test_full)

