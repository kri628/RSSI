import random
import time
import kalman_filter

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd

# import tensorflow.compat.v1 as tfd
# tf.disable_v2_behavior()
import tensorflow as tf
#
# np.random.seed(0)
# tf.set_random_seed(0)
from tensorflow.python.keras import losses
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Dropout

tf.random.set_seed(0)

# ws = 20

# Model configuration
# input_shape = (ws, 8, 1)
# input_shape_1d = (ws, 8)
batch_size = 150
no_epochs = 10
train_test_split = 0.3
# validation_split = 0.2
verbosity = 1
max_norm_value = 2.0

# Sample configuration
num_samples_visualize = 3
noise_factor = 0.3

# noise_factor_arr = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]


def run_kalman_cnn(ws=20, data_num=1):

    filename = "predict/kalman/220923_" + str(ws) + ".csv"
    print(filename, ws)

    # Load data
    data = np.load('./signal_waves_line_' + str(ws) + '.npy')  # (100000, 20, 5)
    x_val_pure = data[:, :, 0]
    y_val_pure = data[:, :, 1:]
    # print(x_val_pure.shape, y_val_pure.shape)
    # (100000, 20) (100000, 20, 8)

    ## data load
    if data_num == 1:
        data = np.load('dataset/rssi_data_full_' + str(ws) + '.npz')
    else:
        data = np.load('model/rssi_ae_cnn_data_full_' + str(ws) + '.npz')

    x_train, x_valid, x_test = data['x_train'], data['x_valid'], data['x_test']
    y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']
    # print(x_train.shape, x_valid.shape, x_test.shape)  # (360746, 20, 8) (72150, 20, 8) (83960, 20, 8)
    # print(y_train.shape, y_valid.shape, y_test.shape)  # (374746, 1) (74950, 1) (89560, 1)

    # x_train = x_train.reshape((-1, ws, 8, 1))
    # x_valid = x_valid.reshape((-1, ws, 8, 1))
    # x_test = x_test.reshape((-1, ws, 8, 1))

    # encoded_train = autoencoder.encoder(x_train).numpy()
    # encoded_valid = autoencoder.encoder(x_valid).numpy()
    # encoded_test = autoencoder.encoder(x_test).numpy()
    #
    # decoded_train = autoencoder.decoder(encoded_train).numpy()
    # decoded_valid = autoencoder.decoder(encoded_valid).numpy()
    # decoded_test = autoencoder.decoder(encoded_test).numpy()

    filtered_train = np.array(kalman_filter.exe_kalman(x_train, ws))
    filtered_valid = np.array(kalman_filter.exe_kalman(x_valid, ws))
    filtered_test = np.array(kalman_filter.exe_kalman(x_test, ws))
    print(filtered_train.shape)     # (360746, 20, 8)

    filtered_train = filtered_train.reshape((-1, ws, 8, 1))
    filtered_valid = filtered_valid.reshape((-1, ws, 8, 1))
    filtered_test = filtered_test.reshape((-1, ws, 8, 1))


    # data graph plot
    for i in range(1):
        random_index = random.randint(0, 45100)
        pred_input = x_train[random_index]
        pred = filtered_train[random_index]

        plt.figure(figsize=(10, 6))
        ax0 = plt.subplot(121)
        ax0.set_ylim([0, 1])
        ax1 = plt.subplot(122)
        ax1.set_ylim([0, 1])

        x_len = np.arange(ws)
        for j in range(8):
            ax0.plot(x_len, pred_input[:, j], label='rssi' + str(j + 1))
            ax1.plot(x_len, pred[:, j], label='pred' + str(j + 1))
        ax0.set_title('Noisy rssi')
        ax1.set_title('Denoised rssi')

        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.show()

    # print(x_train.shape, encoded_train.shape)  # (210741, 20, 8, 1) (210741, 5, 2, 8)

    # encoded_train = encoded_train.reshape((encoded_train.shape[0], -1))
    # encoded_valid = encoded_valid.reshape((encoded_valid.shape[0], -1))
    # encoded_test = encoded_test.reshape((encoded_test.shape[0], -1))
    # print(encoded_train.shape)  # (210741, 80)

    model = Sequential()
    # model.add(Flatten())

    model.add(layers.Input(shape=(ws, 8, 1)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))  # 80
    # model.add(Dropout(0.25))
    # model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(300, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(200, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(150, activation='relu'))
    # model.add(Dense(100, activation='relu'))

    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    model.add(layers.Dense(150, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # early_stop = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(filtered_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_data=(filtered_valid, y_valid),
                        # callbacks=[early_stop],
                        # shuffle=False,
                        verbose=1)

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right')
    # plt.show()

    plt.figure()
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(loc='upper right')
    # plt.show()

    # Boxplot
    y_pred = model.predict(filtered_test)
    # print(y_pred.shape, y_test.shape)  # (89560, 150) (89560, 1)
    # print("test loss, acc: ", model.evaluate(encoded_test, y_test))
    # test loss, acc:  [1.1643580198287964, 0.7951429486274719]

    # y_test = y_test.reshape((1, -1))[0]
    # print(len(y_pred), y_pred[0])
    # print(y_pred.shape, y_test.shape)

    # tp_acc = []
    # for i in range(len(y_pred)):
    #     tp_acc.append(y_pred[i][y_test[i]])
    # # print(np.array(tp_acc).shape, tp_acc[:1])
    #
    # acc_df = pd.DataFrame({'cell': y_test,
    #                        'acc': tp_acc})
    # # print(acc_df.tail())
    # plt.figure(figsize=(24, 12))
    # sns.boxplot(x='cell', y='acc', data=acc_df)
    # plt.tight_layout()
    # # plt.show()
    #
    # # print(acc_df.groupby(['cell'], as_index=False).mean())
    #
    pred_df = pd.DataFrame(y_pred)
    pred_df['cell'] = y_test
    print(pred_df.tail())

    # pred_df.to_csv(filename, mode='w')
    # print("saved", filename)

    return history


if __name__ == '__main__':
    start = time.time()

    # for i in range(4):
    #     for j in range(len(noise_factor_arr)):
    #         run_cnn(ws=50, data_num=1)

    run_kalman_cnn(ws=20, data_num=1)

    print(time.time() - start, "sec")


