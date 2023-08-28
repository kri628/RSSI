import math

import numpy as np
import pandas as pd
import trilateration
import rssi_to_distance


def euclid_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)


filepath = './dataset/rssi_all.csv'

df = pd.read_csv(filepath)
print(df.head())

rssi_df = df[['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8']]
label_df = df[['x', 'y']]

rssi = rssi_df.to_numpy()
cell = label_df.to_numpy()
label = cell * 0.2 + 0.1
print(rssi.shape, label.shape)
# (540856, 8) (540856, 2)

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
print("ade:", np.mean(distance_arr) * 100 / 20)
