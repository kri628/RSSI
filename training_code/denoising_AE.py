import random
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
tf.random.set_seed(0)

# np.random.seed(0)
# tf.set_random_seed(0)
# from tensorflow.python.keras.constraints import max_norm


from tensorflow.python.keras import losses, Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose, Flatten
# from keras_flops import get_flops



# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


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
num_samples_visualize = 1
noise_factor = 0.3

noise_factor_arr = [
                    # 0.01, 0.03, 0.05,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    # 0.7, 0.9
                    ]

linestyle = ['solid', 'dashed', 'dotted', 'dashdot']
color = ['black', 'gray']


def run_DAE(ws=20, noise_number=0, noise_f=0.01, gaus_sigma=0.5, data_num=1, file_save=False, ap=8):
    filename = f"predict/dae_predict/220920_noise{noise_number}_{str(noise_f).replace('.', '')}_{ws}_ap{ap}.csv"
    print(filename)

    # Load data
    data = np.load('./signal_waves_line_' + str(ws) + '.npy')  # (100000, 20, 5)
    x_val_pure = data[:, :, 0]
    y_val_pure = data[:, :, 1:ap+1]

    print(x_val_pure.shape, y_val_pure.shape)
    # (100000, 20) (100000, 20, 8)

    # Add noise to data
    noisy_samples = []
    for i in range(0, len(x_val_pure)):
        # if i % 100 == 0:
        #     print(i)
        pure = np.array(y_val_pure[i])

        if noise_number == 0:
            noise = np.random.normal(0, 1, pure.shape)
            # noise_factor = noise_f
            # noise1 = np.random.binomial(1, 0.3, pure.shape)
            # signal = pure + noise_f * noise
            signal = pure * (1 - noise_f) + noise_f * noise

        elif noise_number == 1:
            # noise 1
            noise = np.random.normal(0, gaus_sigma, pure.shape)
            # k = noise_f
            noise_factor = (1 - pure) * noise_f * 0.7 + 0.00000001
            # signal = pure + noise_factor * noise
            signal = pure * (1 - noise_factor) + noise_factor * noise

        elif noise_number == 2:
            # noise 2
            noise = np.random.normal(0, gaus_sigma, pure.shape)
            # noise_factor = 0.05
            rand = np.random.rand(pure.shape[0], pure.shape[1])
            p = pure
            f = np.where(rand >= p, 1, 0)
            # signal = pure + noise * f * noise_f
            signal = pure * (1 - noise_f) + noise * f * noise_f

        elif noise_number == 3:
            # noise 3
            sigma = (1 - pure) * 0.3
            noise = np.random.normal(0, sigma, pure.shape)
            # noise_factor = 0.01
            # signal = pure + noise * noise_f
            signal = pure * (1 - noise_f) + noise * noise_f

        else:
            print("noise number error")
            return

        sample = np.concatenate((np.array(x_val_pure[i]).reshape((ws, 1)), signal), axis=1)
        noisy_samples.append(sample)

    noisy_samples = np.array(noisy_samples)
    x_val_noisy, y_val_noisy = noisy_samples[:, :, 0], noisy_samples[:, :, 1:]

    # Reshape data
    y_val_noisy_r = []
    y_val_pure_r = []
    for i in range(0, len(y_val_noisy)):
        noisy_sample = y_val_noisy[i]
        pure_sample = y_val_pure[i]
        noisy_sample = (noisy_sample - np.min(noisy_sample)) / (np.max(noisy_sample) - np.min(noisy_sample))
        # pure_sample = (pure_sample - np.min(pure_sample)) / (np.max(pure_sample) - np.min(pure_sample))
        y_val_noisy_r.append(noisy_sample)
        y_val_pure_r.append(pure_sample)
    y_val_noisy_r = np.array(y_val_noisy_r)
    y_val_pure_r = np.array(y_val_pure_r)

    noisy_input = y_val_noisy_r.reshape((y_val_noisy_r.shape[0], y_val_noisy_r.shape[1], y_val_noisy_r.shape[2], 1))
    pure_input = y_val_pure_r.reshape((y_val_pure_r.shape[0], y_val_pure_r.shape[1], y_val_pure_r.shape[2], 1))
    #
    # noisy_input = y_val_noisy_r.reshape((y_val_noisy_r.shape[0], y_val_noisy_r.shape[1], y_val_noisy_r.shape[2]))
    # pure_input = y_val_pure_r.reshape((y_val_pure_r.shape[0], y_val_pure_r.shape[1], y_val_pure_r.shape[2]))

    print(noisy_input.shape, pure_input.shape)
    # (100000, 20, 8, 1) (100000, 20, 8, 1)

    # Train/test split
    percentage_training = math.floor((1 - train_test_split) * len(noisy_input))
    noisy_input, noisy_input_test = noisy_input[:percentage_training], noisy_input[percentage_training:]
    pure_input, pure_input_test = pure_input[:percentage_training], pure_input[percentage_training:]

    class Denoise(Model):
        def __init__(self):
            super(Denoise, self).__init__()
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=(ws, ap, 1)),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            ])

            self.decoder = tf.keras.Sequential([
                layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
                layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
                layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = Denoise()

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # flops = get_flops(autoencoder, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")

    dae_start = time.time()
    autoencoder.fit(noisy_input, pure_input,
                    epochs=10,
                    shuffle=True,
                    validation_split=0.2,
                    # batch_size=1,
                    verbose=1)
    print("dae time:", time.time() - dae_start, "sec")
    # dae time: 192.50608611106873 sec

    autoencoder.summary()
    # encoded_imgs = autoencoder.encoder(noisy_input_test).numpy()
    # decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    # Generate reconstructions
    num_reconstructions = 1
    samples = noisy_input_test[:num_reconstructions]
    # samples = noisy_input_test
    reconstructions = autoencoder.call(samples)

    # np.save('npy/dae_reconstructions' + str(ws) + '.npy', np.array([pure_input_test, noisy_input_test, reconstructions]))

    # print(samples.shape, reconstructions.shape)
    # (3, 20, 8, 1) (3, 20, 8, 1)

    # Plot reconstructions
    for i in np.arange(0, num_reconstructions):
        # Prediction index
        prediction_index = i + percentage_training
        # Get the sample and the reconstruction
        original = y_val_noisy[prediction_index]
        pure = y_val_pure[prediction_index]
        reconstruction = np.array(reconstructions[i])
        # Matplotlib preparations
        plt.figure(figsize=(12, 6))
        ax0 = plt.subplot(131)
        ax0.set_ylim([0, 1])
        ax1 = plt.subplot(132)
        ax1.set_ylim([0, 1])
        ax2 = plt.subplot(133)
        ax2.set_ylim([0, 1])
        # Plot sample and reconstruciton
        # (100, 4) (100, 4) (100, 4, 1)
        for j in range(ap):
            ax0.plot(original[:, j], color=color[j//4], linestyle=linestyle[j%4], label='AP' + str(j+1))
            ax1.plot(pure[:, j], color=color[j//4], linestyle=linestyle[j%4], label='AP' + str(j+1))
            ax2.plot(reconstruction[:, j], color=color[j//4], linestyle=linestyle[j%4], label='AP' + str(j+1))
        # ax0.set_title('Noisy waveform')
        # ax1.set_title('Pure waveform')
        # ax2.set_title('Denoised waveform')
        # ax0.legend(loc='upper right')
        # ax1.legend(loc='upper right')
        # ax2.legend(loc='upper right')

        ax0.set_xlabel('time(unit: 100msec)')
        ax0.set_ylabel('RSSI(dB)')
        ax1.set_xlabel('time(unit: 100msec)')
        ax1.set_ylabel('RSSI(dB)')
        ax2.set_xlabel('time(unit: 100msec)')
        ax2.set_ylabel('RSSI(dB)')

        plt.tight_layout()
        plt.show()

    # 1750/1750 [==============================] - 7s 4ms/step - loss: 0.0078 - val_loss: 0.0077

    '''data load'''
    if data_num == 1:
        data = np.load('dataset/rssi_data_full_' + str(ws) + '.npz')
    else:
        data = np.load('model/rssi_ae_cnn_data_full_' + str(ws) + '.npz')

    x_train, x_valid, x_test = data['x_train'], data['x_valid'], data['x_test']
    y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']
    # (225241, ws, 8) (75245, ws, 8) (75245, ws, 8)


    if ap == 3:
        x_train = np.stack([x_train[:, :, 0], x_train[:, :, 2], x_train[:, :, 4]], axis=2)
        x_valid = np.stack([x_valid[:, :, 0], x_valid[:, :, 2], x_valid[:, :, 4]], axis=2)
        x_test = np.stack([x_test[:, :, 0], x_test[:, :, 2], x_test[:, :, 4]], axis=2)

    if ap == 4:
        x_train = np.stack([x_train[:, :, 0], x_train[:, :, 2], x_train[:, :, 4], x_train[:, :, 6]], axis=2)
        x_valid = np.stack([x_valid[:, :, 0], x_valid[:, :, 2], x_valid[:, :, 4], x_valid[:, :, 6]], axis=2)
        x_test = np.stack([x_test[:, :, 0], x_test[:, :, 2], x_test[:, :, 4], x_test[:, :, 6]], axis=2)

    if ap == 6:
        x_train = np.stack([x_train[:, :, 0], x_train[:, :, 2], x_train[:, :, 3], x_train[:, :, 4], x_train[:, :, 6], x_train[:, :, 7]], axis=2)
        x_valid = np.stack([x_valid[:, :, 0], x_valid[:, :, 2], x_valid[:, :, 3], x_valid[:, :, 4], x_valid[:, :, 6], x_valid[:, :, 7]], axis=2)
        x_test = np.stack([x_test[:, :, 0], x_test[:, :, 2], x_test[:, :, 3], x_test[:, :, 4], x_test[:, :, 6], x_test[:, :, 7]], axis=2)

    print(x_train.shape, x_valid.shape, x_test.shape)  # (360746, 20, 8) (72150, 20, 8) (83960, 20, 8)
    print(y_train.shape, y_valid.shape, y_test.shape)  # (360746, 1) (72150, 1) (83960, 1)

    x_train = x_train.reshape((-1, ws, ap, 1))
    x_valid = x_valid.reshape((-1, ws, ap, 1))
    x_test = x_test.reshape((-1, ws, ap, 1))

    encoded_train = np.empty((0, ws, ap, 16))
    encoded_valid = np.empty((0, ws, ap, 16))
    encoded_test = np.empty((0, ws, ap, 16))

    size = x_train.shape[0] // 10
    # print(size)
    for i in range(9):
        encoded_train = np.append(encoded_train, autoencoder.encoder(x_train[i * size:(i + 1) * size]).numpy(), axis=0)
        # print(encoded_train.shape)
    encoded_train = np.append(encoded_train, autoencoder.encoder(x_train[9 * size:]).numpy(), axis=0)
    print(encoded_train.shape)

    for i in range(9):
        encoded_valid = np.append(encoded_valid, autoencoder.encoder(x_valid[i * size:(i + 1) * size]).numpy(), axis=0)
        # print(encoded_valid.shape)
    encoded_valid = np.append(encoded_valid, autoencoder.encoder(x_valid[9 * size:]).numpy(), axis=0)
    print(encoded_valid.shape)

    for i in range(9):
        encoded_test = np.append(encoded_test, autoencoder.encoder(x_test[i * size:(i + 1) * size]).numpy(), axis=0)
        # print(encoded_test.shape)
    encoded_test = np.append(encoded_test, autoencoder.encoder(x_test[9 * size:]).numpy(), axis=0)
    print(encoded_test.shape)

    # encoded_train = autoencoder.encoder(x_train).numpy()
    # encoded_valid = autoencoder.encoder(x_valid).numpy()
    # encoded_test = autoencoder.encoder(x_test).numpy()

    decoded_train = np.empty((0, ws, ap, 1))
    decoded_valid = np.empty((0, ws, ap, 1))
    decoded_test = np.empty((0, ws, ap, 1))

    for i in range(9):
        decoded_train = np.append(decoded_train, autoencoder.decoder(encoded_train[i * size:(i + 1) * size]).numpy(), axis=0)
        # print(decoded_train.shape)
    decoded_train = np.append(decoded_train, autoencoder.decoder(encoded_train[9 * size:]).numpy(), axis=0)
    print(decoded_train.shape)

    for i in range(9):
        decoded_valid = np.append(decoded_valid, autoencoder.decoder(encoded_valid[i * size:(i + 1) * size]).numpy(), axis=0)
        # print(decoded_train.shape)
    decoded_valid = np.append(decoded_valid, autoencoder.decoder(encoded_valid[9 * size:]).numpy(), axis=0)
    print(decoded_valid.shape)

    for i in range(9):
        decoded_test = np.append(decoded_test, autoencoder.decoder(encoded_test[i * size:(i + 1) * size]).numpy(), axis=0)
        # print(decoded_train.shape)
    decoded_test = np.append(decoded_test, autoencoder.decoder(encoded_test[9 * size:]).numpy(), axis=0)
    print(decoded_test.shape)

    # decoded_train = autoencoder.decoder(encoded_train).numpy()
    # decoded_valid = autoencoder.decoder(encoded_valid).numpy()
    # decoded_test = autoencoder.decoder(encoded_test).numpy()

    # np.savez('npy/dae_reconstructions_rssi_' + str(ws) + '.npz', x_train=x_train, decoded_train=decoded_train)
    np.savez('npy/dae_decoded_' + str(ws) + '.npz', x_data=np.concatenate([decoded_train, decoded_valid, decoded_test]),
                                                    y_data=np.concatenate([y_train, y_valid, y_test]))


    # data graph plot
    for i in range(2):
        random_index = random.randint(0, 45100)
        pred_input = x_train[random_index]
        pred = decoded_train[random_index]

        plt.figure(figsize=(10, 6))
        ax0 = plt.subplot(121)
        ax0.set_ylim([0, 1])
        ax1 = plt.subplot(122)
        ax1.set_ylim([0, 1])

        x_len = np.arange(ws)
        for j in range(ap):
            ax0.plot(x_len, pred_input[:, j], color=color[j//4], linestyle=linestyle[j%4])
            ax1.plot(x_len, pred[:, j], color=color[j//4], linestyle=linestyle[j%4], label='rssi' + str(j+1))
        ax0.set_title('Noisy RSSI', fontsize=15)
        ax1.set_title('Denoised RSSI', fontsize=15)
        ax0.set_xlabel('time(unit: 100msec)')
        ax0.set_ylabel('RSSI(dB)')
        ax1.set_xlabel('time(unit: 100msec)')
        ax1.set_ylabel('RSSI(dB)')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    # print(x_train.shape, encoded_train.shape)  # (210741, 20, 8, 1) (210741, 5, 2, 8)

    encoded_train = encoded_train.reshape((encoded_train.shape[0], -1))
    encoded_valid = encoded_valid.reshape((encoded_valid.shape[0], -1))
    encoded_test = encoded_test.reshape((encoded_test.shape[0], -1))
    # print(encoded_train.shape)  # (210741, 80)

    '''dnn model'''
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(512, input_dim=encoded_train[0].shape[0], activation='relu'))  # 80
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
    model.add(Dense(150, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # early_stop = EarlyStopping(monitor='val_loss', patience=3)

    cls_start = time.time()
    history = model.fit(encoded_train, y_train,
                        epochs=20,
                        # batch_size=16,
                        validation_data=(encoded_valid, y_valid),
                        # callbacks=[early_stop],
                        shuffle=True,
                        verbose=1)
    print("FCN time:", time.time() - cls_start, "sec")
    # FCN time: 1304.4057590961456 sec
    model.summary()

    # plt.figure()
    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # plt.figure()
    # plt.ylim([0, 1])
    # plt.plot(history.history['accuracy'], label='acc')
    # plt.plot(history.history['val_accuracy'], label='val_acc')
    # plt.legend(loc='upper right')
    # plt.show()

    ### model save
    # autoencoder.save('model/autoencoder.h5')
    # model.save('model/classifier.h5')

    # Boxplot
    y_pred = model.predict(encoded_test)
    # print(y_pred.shape, y_test.shape)  # (89560, 150) (89560, 1)
    print("test loss, acc: ", model.evaluate(encoded_test, y_test))
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
    # print(acc_df.groupby(['cell'], as_index=False).mean())

    pred_df = pd.DataFrame(y_pred)
    pred_df['cell'] = y_test
    print(pred_df.tail())

    if file_save:
        pred_df.to_csv(filename, mode='w')
        print("saved", filename)
    else:
        print("file no save")

    return history


if __name__ == '__main__':
    start = time.time()

    # for i in range(4):
    #     for j in range(len(noise_factor_arr)):
    #         run_DAE(ws=20, noise_number=i, noise_f=noise_factor_arr[j], gaus_sigma=1, data_num=1, file_save=True)

    # for j in range(3):
    #     run_DAE(ws=20, noise_number=0, noise_f=noise_factor_arr[j+2], gaus_sigma=0.5, data_num=1, file_save=True)
    #
    # for i in range(4):
    #     run_DAE(ws=20, noise_number=i, noise_f=0.3, gaus_sigma=0.5, data_num=1, file_save=True, ap=6)

    # for i in (30, 40):
    #     run_DAE(ws=i, noise_number=2, noise_f=0.3, gaus_sigma=0.5, data_num=1, file_save=True)

    run_DAE(ws=20, noise_number=2, noise_f=0.3, gaus_sigma=0.5, data_num=1, file_save=False)

    print((time.time() - start)/60, "min")
