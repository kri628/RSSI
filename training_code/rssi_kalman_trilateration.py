import random
import time
import kalman_filter

import matplotlib.pyplot as plt
import numpy as np

import math
import trilateration
import rssi_to_distance

import joblib

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


def euclid_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)


def run_kalman_cnn(ws=20, data_num=1):

    filename = "predict/kalman/trilateration_" + str(ws) + ".csv"
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

    x_data = np.concatenate([x_train, x_valid, x_test])
    y_data = np.concatenate([y_train, y_valid, y_test])

    cell = np.array([[x[0]//10, x[0]%10] for x in y_data])
    label = cell * 0.2 + 0.1
    print(label[-5:])
    print(x_data.shape, label.shape)
    # (516856, 20, 8) (516856, 2)

    filtered_x = np.array(kalman_filter.exe_kalman(x_data, ws))
    print(filtered_x.shape)
    # (516856, 20, 8)

    # # data graph plot
    # for i in range(1):
    #     random_index = random.randint(0, 45100)
    #     pred_input = x_train[random_index]
    #     pred = filtered_x[random_index]
    #
    #     plt.figure(figsize=(10, 6))
    #     ax0 = plt.subplot(121)
    #     ax0.set_ylim([0, 1])
    #     ax1 = plt.subplot(122)
    #     ax1.set_ylim([0, 1])
    #
    #     x_len = np.arange(ws)
    #     for j in range(8):
    #         ax0.plot(x_len, pred_input[:, j], label='rssi' + str(j + 1))
    #         ax1.plot(x_len, pred[:, j], label='pred' + str(j + 1))
    #     ax0.set_title('Noisy rssi')
    #     ax1.set_title('Denoised rssi')
    #
    #     plt.tight_layout()
    #     plt.legend(loc='upper right')
    #     plt.show()

    # print(x_train.shape, encoded_train.shape)  # (210741, 20, 8, 1) (210741, 5, 2, 8)

    # encoded_train = encoded_train.reshape((encoded_train.shape[0], -1))
    # encoded_valid = encoded_valid.reshape((encoded_valid.shape[0], -1))
    # encoded_test = encoded_test.reshape((encoded_test.shape[0], -1))
    # print(encoded_train.shape)  # (210741, 80)

    rssi = []
    for i in range(len(filtered_x)):
        arr = []
        for j in range(8):
            arr.append(np.mean(filtered_x[i, :, j]))
        rssi.append(arr)
    rssi = np.array(rssi)

    print(rssi.shape, label.shape)
    print(rssi[:5])
    # (516856, 8) (516856, 2)

    ## Scaler
    scaler = joblib.load('model/rssi_dae_mms_full_' + str(ws) + '.pkl')
    rssi = scaler.inverse_transform(rssi)
    print(rssi[:5])

    '''trilateration'''
    app = rssi_to_distance.App(rows=2, txpower=-50)
    cal_dis = np.vectorize(app.calculate_distance)
    distance = cal_dis(rssi)
    print(distance)

    pred_label = []

    for i in range(len(rssi)):
        try:
            x, y = trilateration.exe_trilateration(distance_row=distance[i])
            pred_label.append([x, y])

        except ZeroDivisionError:
            pred_label.append([-1, -1])

    # print(pred_label)

    fail_cnt = 0
    distance_arr = []
    # x_dis, y_dis = [], []
    pred_cell = []
    acc_cnt = 0

    for i in range(len(pred_label)):
        if pred_label[i][0] == -1:
            fail_cnt += 1
        else:
            distance_err = euclid_distance(pred_label[i], label[i])
            distance_arr.append(distance_err)
            print(pred_label[i], label[i], distance_err)
            # x_dis.append(abs(pred_label[i][0] - label[i][0]))
            # y_dis.append(abs(pred_label[i][1] - label[i][1]))
            pred_cell = [np.trunc(pred_label[i][0] * 5), np.trunc(pred_label[i][1] * 5)]
            print(pred_cell, cell[i])

            if pred_cell[0] == cell[i][0] and pred_cell[1] == cell[i][1]:
                acc_cnt += 1

    # x_mean = np.mean(x_dis)
    # y_mean = np.mean(y_dis)
    # print(x_mean, y_mean)
    #
    # x_err = x_mean / 3.0
    # y_err = y_mean / 2.0
    # print(x_err, y_err)
    #
    # err_rate = np.mean([x_err, y_err])
    # print(err_rate)
    #
    # acc = 1 / err_rate
    # print(acc)

    print("total", len(rssi))
    print("failed", fail_cnt)

    print(acc_cnt)
    print("acc:", acc_cnt / (len(rssi) - fail_cnt))
    print("ade:", np.mean(distance_arr) * 100)
    print("std:", np.std(distance_arr) * 100)
    print(len(distance_arr))


if __name__ == '__main__':
    start = time.time()

    # for i in range(4):
    #     for j in range(len(noise_factor_arr)):
    #         run_cnn(ws=50, data_num=1)

    run_kalman_cnn(ws=20, data_num=1)

    print(time.time() - start, "sec")


