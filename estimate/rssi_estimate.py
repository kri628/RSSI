import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def distance(cell1, cell2):
    dx = cell1 / 10 - cell2 / 10
    dy = cell1 % 10 - cell2 % 10
    return math.sqrt(dx**2 + dy**2)


def estimate(df, graph_show=False):
    # df = pd.read_csv('predict/dae/220920_noise0_001_20.csv')
    # print(df)
    arr = df.to_numpy()
    # print(arr.shape)     # (83960, 152)
    cnt = df['cell'].value_counts().sort_index().to_numpy()
    # print(cnt.shape, cnt[:5])

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
    ade_cm = np.round(ade * 20, 2)
    # print("acc :", acc)
    # print("ade :", ade, '/', ade_cm, "cm", end='\n')

    # print(acc_map)

    # acc_map = np.floor(100 * np.array(acc_map)) / 100   # 소수점 버림
    # err_map = np.floor(100 * np.array(err_map)) / 100

    # print(acc_map)

# ## threshold
#     err_map = np.array(err_map)
#     ade_arr = []
#     for e in range(11):
#         a = err_map[err_map <= e]
#         print(len(a))
#         ade_arr.append(np.mean(a))
#     print(ade_arr)

    # plt.figure(figsize=(10, 4))
    # plt.plot(np.arange(11), ade_arr, color='k', marker='o')
    # plt.ylabel('ADE', fontsize=15)
    # plt.xlabel('Unit of distance between neighbor cells', fontsize=15)
    # plt.tight_layout()
    # plt.show()

    if graph_show:
        plt.figure(figsize=(6, 8), dpi=200)
        sns.set(rc={'figure.figsize': (6, 8)})
        sns.heatmap(err_map, annot=True, fmt='.2f', cmap='Blues', annot_kws={'size': 12}, cbar=False)
        plt.title('Average Distance Error(cell)', fontsize=15)
        plt.ylabel('y-cell', fontsize=13)
        plt.xlabel('x-cell', fontsize=13)
        plt.tight_layout()
        plt.show()
        # plt.savefig('predict/dae/220723_ade.png')

        plt.figure(figsize=(6, 8), dpi=200)
        sns.set(rc={'figure.figsize': (6, 8)})
        sns.heatmap(acc_map, annot=True, fmt='.2f', cmap='Greens', annot_kws={'size': 12}, cbar=False)
        plt.title('Classification Accuracy', fontsize=15)
        plt.ylabel('y-cell', fontsize=13)
        plt.xlabel('x-cell', fontsize=13)
        plt.tight_layout()
        plt.show()
        # plt.savefig('predict/dae/220723_acc.png')

    return acc, ade, ade_cm


def main():
    acc, ade, ade_cm = estimate(pd.read_csv('predict/dae_predict/220920_noise2_03_20.csv'), graph_show=True)
    print("acc :", acc)
    print("ade :", ade, '/', ade_cm, "cm", end='\n')


if __name__ == "__main__":
    main()
