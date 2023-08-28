import glob
import re
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def distance(cell1, cell2):
    dx = cell1 / 10 - cell2 / 10
    dy = cell1 % 10 - cell2 % 10
    return math.sqrt(dx ** 2 + dy ** 2)

'''
date = '220920'
# files = glob.glob('predict/dae/' + date + '_noise*[0-9][0-9].csv')   # data1

files = glob.glob('predict/dae/' + date + '*_20.csv')   # data1
# files = glob.glob('predict/dae/220821_noise*data2.csv')   # data2
print(files)
print(len(files))

acc_arr = [[0]*8 for _ in range(4)]
ade_arr = [[0]*8 for _ in range(4)]

for file in files:

    df = pd.read_csv(file)
    cnt = df['cell'].value_counts().sort_index().to_numpy()
    arr = df.to_numpy()

    noise = int(file[24])
    alpha = file.split('_')[2]
    # alpha = file.split('_')[2].split('.')[0]

    acc_map = [[0] * 10 for _ in range(15)]
    rank_map = np.array([[200.0] * 10 for _ in range(15)])
    err_map = [[0] * 10 for _ in range(15)]
    max_err_map = [[0] * 10 for _ in range(15)]

    for i in range(len(arr)):
        a = arr[i][1:150]
        pred_cell = np.argmax(a)
        cell = arr[i][151]
        # print(pred_cell, cell)
        x, y = int(cell // 10), int(cell % 10)

        if pred_cell == cell:
            acc_map[x][y] += 1

        err_map[x][y] += distance(pred_cell, cell)

    for i in range(150):
        x, y = i // 10, i % 10
        acc_map[x][y] /= cnt[i]
        err_map[x][y] /= cnt[i]

    acc = np.mean(acc_map)
    ade = np.mean(err_map)
    # ade = np.round(np.mean(err_map) * 20, 2)

    # print(noise, alpha)
    # print("acc :", acc)
    # print("ade :", np.mean(err_map))
    # print("ade :", ade)
    # print()

    if alpha == '001':
        ade_arr[noise][0] = ade
        acc_arr[noise][0] = acc
    elif alpha == '003':
        ade_arr[noise][1] = ade
        acc_arr[noise][1] = acc
    elif alpha == '005':
        ade_arr[noise][2] = ade
        acc_arr[noise][2] = acc
    elif alpha == '01':
        ade_arr[noise][3] = ade
        acc_arr[noise][3] = acc
    elif alpha == '03':
        ade_arr[noise][4] = ade
        acc_arr[noise][4] = acc
    elif alpha == '05':
        ade_arr[noise][5] = ade
        acc_arr[noise][5] = acc
    elif alpha == '07':
        ade_arr[noise][6] = ade
        acc_arr[noise][6] = acc
    elif alpha == '09':
        ade_arr[noise][7] = ade
        acc_arr[noise][7] = acc



    # print(acc_map)
    #
    # acc_map = np.floor(100 * np.array(acc_map)) / 100   # 소수점 버림
    # err_map = np.floor(100 * np.array(err_map)) / 100

print(np.array(acc_arr))
print(np.array(ade_arr))

x = ['0.01', '0.03', '0.05', '0.1', '0.3', '0.5', '0.7', '0.9']
plt.figure(figsize=(12, 5))
plt.ylim([1.0, 3.0])
plt.plot(x, ade_arr[0], label='gaussian')
plt.plot(x, ade_arr[1], label='noise1')
plt.plot(x, ade_arr[2], label='noise2')
plt.plot(x, ade_arr[3], label='noise3')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
'''

date = '220920'

files = []
files.append(glob.glob('predict/dae/' + date + '_noise0_*_10.csv'))   # data1
files.append(glob.glob('predict/dae/' + date + '_noise0_*_20.csv'))   # data1
files.append(glob.glob('predict/dae/' + date + '_noise0_*_50.csv'))   # data1

ade_arr = [[0]*8 for _ in range(3)]

for k in range(3):
    for file in files[k]:

        df = pd.read_csv(file)
        cnt = df['cell'].value_counts().sort_index().to_numpy()
        arr = df.to_numpy()

        ws = int(file[-6:-4])
        alpha = file.split('_')[2]

        acc_map = [[0] * 10 for _ in range(15)]
        rank_map = np.array([[200.0] * 10 for _ in range(15)])
        err_map = [[0] * 10 for _ in range(15)]
        max_err_map = [[0] * 10 for _ in range(15)]

        for i in range(len(arr)):
            a = arr[i][1:150]
            pred_cell = np.argmax(a)
            cell = arr[i][151]
            # print(pred_cell, cell)
            x, y = int(cell // 10), int(cell % 10)

            if pred_cell == cell:
                acc_map[x][y] += 1

            err_map[x][y] += distance(pred_cell, cell)

        for i in range(150):
            x, y = i // 10, i % 10
            acc_map[x][y] /= cnt[i]
            err_map[x][y] /= cnt[i]

        acc = np.mean(acc_map)
        ade = np.mean(err_map)
        # ade = np.round(np.mean(err_map) * 20, 2)

        # print(ws, alpha)
        # print("acc :", acc)
        # print("ade :", ade)
        # print()

        if alpha == '001':
            ade_arr[k][0] = ade
            acc_map[k][0] = acc
        elif alpha == '003':
            ade_arr[k][1] = ade
            acc_map[k][1] = acc
        elif alpha == '005':
            ade_arr[k][2] = ade
            acc_map[k][2] = acc
        elif alpha == '01':
            ade_arr[k][3] = ade
            acc_map[k][3] = acc
        elif alpha == '03':
            ade_arr[k][4] = ade
            acc_map[k][4] = acc
        elif alpha == '05':
            ade_arr[k][5] = ade
            acc_map[k][5] = acc
        elif alpha == '07':
            ade_arr[k][6] = ade
            acc_map[k][6] = acc
        elif alpha == '09':
            ade_arr[k][7] = ade
            acc_map[k][7] = acc

print(ade_arr)

x = ['0.01', '0.03', '0.05', '0.1', '0.3', '0.5', '0.7', '0.9']
plt.figure(figsize=(10, 4))
# plt.ylim([1.0, 3.0])
plt.plot(x, ade_arr[0], label='window size=10', color='gray', linestyle='solid')
plt.plot(x, ade_arr[1], label='window size=20', color='gray', linestyle='dashed')
plt.plot(x, ade_arr[2], label='window size=50', color='gray', linestyle='dotted')
plt.xlabel('noise ratio', fontsize=15)
plt.ylabel('ADE', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.tight_layout()
plt.show()

