import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats


def distance(cell1, cell2, size):
    dx = cell1 / size - cell2 / size
    dy = cell1 % size - cell2 % size
    return math.sqrt(dx**2 + dy**2)


def cal_cdf(file, grid=1, show=False):
    # file = 'predict/dae_predict/220920_noise2_03_20.csv'
    # file = 'predict/kalman/220923_20.csv'
    df = pd.read_csv(file)
    print(df)

    cell_cm = grid * 20
    x_cell = math.ceil(15 / grid)
    y_cell = math.ceil(10 / grid)
    print("grid", x_cell, y_cell)

    arr = df.to_numpy()
    # print(arr.shape)     # (83960, 152)
    # cnt = df['cell'].value_counts().sort_index().to_numpy()
    # print(cnt.shape, cnt[:5])

    err = []

    for i in range(len(arr)):
        a = arr[i][1:arr.shape[1]-1]
        pred_cell = np.argmax(a)
        cell = arr[i][arr.shape[1]-1]
        # print(pred_cell, cell)
        # x, y = int(cell // 10), int(cell % 10)

        err.append(distance(pred_cell, cell, y_cell) * cell_cm)

    # print(len(err), max(err), min(err))
    # print(np.std(err))

    cdf = stats.norm.cdf(sorted(err), np.mean(err), np.std(err))

    if show:
        plt.plot(sorted(err), cdf, color="Black", label="Cultivar_A")
        # plt.xlim([0, 30])
        plt.ylim([0, 1])
        plt.legend()
        plt.xlabel('Average Distance Error (cm)', size=15)
        # plt.ylabel("Frequency", size=15)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.rcParams["figure.figsize"] = [7, 5]  # 가로, 세로 인치 조정
        plt.rcParams["figure.dpi"] = 500   # 해상도 조정
        plt.show()

    return cdf, sorted(err)
