import math

import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import natsort
import seaborn as sns


path = "./predict/dae/*"
file_list = glob.glob(path)
file_list = natsort.natsorted(file_list)


def distance(cell1, cell2):
    dx = cell1 / 10 - cell2 / 10
    dy = cell1 % 10 - cell2 % 10
    return math.sqrt(dx**2 + dy**2)


def estimate(flist):
    acc_map = [[-100.0] * 10 for _ in range(15)]
    rank_map = [[200.0] * 10 for _ in range(15)]
    err_map = [[-10.0] * 10 for _ in range(15)]
    max_err_map = [[-10.0] * 10 for _ in range(15)]

    name = np.arange(150)
    cnt, dis, t5_p = [], [], []
    for file in flist:
        df = pd.read_csv(file, engine='python', header=1, names=name)
        df = df.to_numpy()
        c, d, t5, rank = 0, 0, 0, 0
        maxd, sumd = 0, 0

        s = file.split('/')[-1]
        s = s[7:]
        cell = int(re.sub(r'[^0-9]', '', s))
        x, y = cell // 10, cell % 10

        for row in df:
            idx = row.argsort()[:][::-1]
            rank += np.where(idx == cell)[0][0]

            if cell == idx[0]:
                c += 1
            d = distance(cell, idx[0])
            sumd += d
            if maxd < d:
                maxd = d

            if cell in idx[:5]:
                t5 += 1

        cnt.append(c)
        dis.append(sumd / len(df))
        t5_p.append(t5)

        acc_map[x][y] = c / len(df) * 100
        rank_map[x][y] = rank / len(df)
        err_map[x][y] = sumd / len(df)
        max_err_map[x][y] = maxd

    print("cnt:", np.array(cnt))
    dis = np.array(dis)
    print("accuracy:", np.mean(cnt), "dis:", dis)
    ade = np.mean(dis)
    print("average distance error:", ade)
    print("top-5 cnt:", t5_p)

    sns.set(rc={'figure.figsize': (8, 6)})
    sns.heatmap(acc_map, annot=True, fmt='.1f', center=0, cmap='Greens')
    plt.title('accuracy')
    plt.show()

    sns.set(rc={'figure.figsize': (8, 6)})
    sns.heatmap(err_map, annot=True, fmt='.2f', center=0, cmap='Blues')
    plt.title('distance error')
    plt.show()

    sns.set(rc={'figure.figsize': (8, 6)})
    sns.heatmap(rank_map, annot=True, fmt='.2f', vmax=149, vmin=0, cmap='summer')
    plt.title('rank')
    plt.show()

    sns.set(rc={'figure.figsize': (8, 6)})
    sns.heatmap(max_err_map, annot=True, fmt='.2f', center=0, cmap='Blues')
    plt.title('max error')
    plt.show()

    ### distance error 기준 5개 좌표
    distace_err_list = np.array(err_map).flatten()
    err_index = distace_err_list.argsort()[:][::-1]
    # print(err_index)

    for err_cell in err_index[:5]:
        df = pd.read_csv('./predict/220411/220411_ae_cnn_' + str(err_cell) + '.csv', engine='python', header=1, names=name)
        df = df.to_numpy()

        predict_map = [[0] * 10 for _ in range(15)]

        for row in df:
            idx = row.argsort()[:][::-1]
            cell = idx[0]
            x, y = cell // 10, cell % 10

            predict_map[x][y] += 1

        sns.set(rc={'figure.figsize': (8, 6)})
        sns.heatmap(predict_map, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.title(str(err_cell) + ' predict')
        plt.show()

    ### max error 기준 5개 좌표
    max_err_list = np.array(err_map).flatten()
    max_index = max_err_list.argsort()[:][::-1]
    # print(max_index)

    for max_cell in max_index[:5]:
        df = pd.read_csv('./predict/220411/220411_ae_cnn_' + str(max_cell) + '.csv', engine='python', header=1,
                         names=name)
        df = df.to_numpy()

        predict_map = [[0] * 10 for _ in range(15)]

        for row in df:
            idx = row.argsort()[:][::-1]
            cell = idx[0]
            x, y = cell // 10, cell % 10

            predict_map[x][y] += 1

        sns.set(rc={'figure.figsize': (8, 6)})
        sns.heatmap(predict_map, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.title(str(max_cell) + ' predict')
        plt.show()

    print("\n\n")


estimate(file_list)

# x = np.arange(12)
# plt.figure(figsize=(12, 6))
# ax0 = plt.subplot(121)
# ax0.set_ylim([0, 105])
# ax1 = plt.subplot(122)
# ax1.set_ylim([0, 105])
#
# ax0.plot(x, cnt[12:], 'r^-', label='CNN')
# ax1.plot(x, t5_p[12:], 'r^--', label='CNN')
# ax0.plot(x, cnt[:12], 'b*-', label='AE+CNN')
# ax1.plot(x, t5_p[:12], 'b*--', label='AE+CNN')
# ax0.set_title('Top-1 accuracy')
# ax1.set_title('Top-5 accuracy')
#
# ax1.legend(loc='upper right')
# plt.tight_layout()
# # plt.legend(loc='upper right')
# plt.show()
#
# # xlabel = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
# # plt.xticks(x, xlabel)
# plt.figure(figsize=(12, 8))
# ax0 = plt.subplot(211)
# ax0.set_ylim([0, 105])
# ax1 = plt.subplot(212)
# ax1.set_ylim([0, 105])
#
# linewidth = 0.4
# ax0.bar(x, cnt[12:], label='CNN', width=linewidth, color='r', alpha=0.5)
# ax0.bar(x+linewidth, cnt[:12], label='AE+CNN', width=linewidth, color='b', alpha=0.5)
# ax1.bar(x, t5_p[12:], label='CNN', width=linewidth, color='r', alpha=0.5)
# ax1.bar(x+linewidth, t5_p[:12], label='AE+CNN', width=linewidth, color='b', alpha=0.5)
#
# ax0.legend(loc='upper right')
# ax0.set_title('Top-1 accuracy')
# ax1.set_title('Top-5 accuracy')
#
# ax0.set_xlabel("test point")
# ax0.set_ylabel("accuracy")
#
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(12, 6))
# plt.ylim([0, 105])
#
# linewidth = 0.4
# plt.plot(x, cnt[12:], 'r^-', label='CNN')
# plt.bar(x, cnt[:12], label='AE+CNN', width=linewidth, color='b', alpha=0.5)
#
# plt.legend(loc='upper right')
# plt.title('Accuracy for each test point')
#
# plt.xlabel("test point")
# plt.ylabel("accuracy")
#
# xlabel = ['0', '5', '9', '50', '55', '59', '90', '95', '99', '140', '145', '149']
# plt.xticks(x, xlabel)
#
# plt.tight_layout()
# plt.show()
