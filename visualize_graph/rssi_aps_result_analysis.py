import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import rssi_estimate

# filelist = glob.glob('predict/dae_predict/220920_*.csv')

aps = ('3\nAP(1, 3, 5)', '4\nAP(1, 3, 5, 7)', '6\nAP(1, 3, 4, 5, 7, 8)', '8\nAP(1-8)')
colors = ['red', 'green', 'blue', 'darkorchid']
noise = ['Gaussian', 'Proportional factor', 'Occurrence Probability', 'Standard deviation']
marker = ['.', '^', 'p', '*']

filelist, acc_list, ade_list = [], [], []
for noise_number in range(4):
    name_arr = []
    acc_arr = []
    ade_arr = []
    for ap in (3, 4, 6, 8):
        if ap == 8:
            name_inner = sorted(glob.glob(f'predict/dae_predict/220920_noise{noise_number}_03_20.csv'))
        else:
            name_inner = sorted(glob.glob(f'predict/dae_predict/220920_noise{noise_number}_03_20_ap{ap}.csv'))

        name_arr.append(name_inner)

        acc_inner, ade_inner = [], []
        for filename in name_inner:
            df = pd.read_csv(filename)
            acc, ade, ade_cm = rssi_estimate.estimate(df, False)
            acc_inner.append(acc)
            ade_inner.append(ade_cm)
        acc_arr.append(acc_inner)
        ade_arr.append(ade_inner)

    filelist.append(name_arr)
    acc_list.append(acc_arr)
    ade_list.append(ade_arr)

# print(np.array(filelist))

filelist = np.array(filelist)
acc_list = np.array(acc_list)
ade_list = np.array(ade_list)
print(filelist, acc_list, ade_list, sep='\n')
# print(filelist.shape)
# print(filelist[:, 0])

# x = np.arange(4)
# figsize = (6, 4)

# plt.figure(figsize=figsize, dpi=200)
# plt.bar(x, ade_list[:, 0, 0], color=colors)
# plt.bar(x, ade_list[:, 1, 0], color=colors)
# plt.bar(x, ade_list[:, 2, 0], color=colors)
# plt.bar(x, ade_list[:, 3, 0], color=colors)
# plt.ylabel('Average Distance Error (cm)', fontsize=13)

# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
bar_width = 0.2

# 연도가 4개이므로 0, 1, 2, 3 위치를 기준으로 삼음
index = np.arange(4) - 0.1

# 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
b1 = plt.bar(index, ade_list[0, :, 0], bar_width, alpha=0.5, color=colors[0], label=noise[0])

b2 = plt.bar(index + bar_width, ade_list[1, :, 0], bar_width, alpha=0.5, color=colors[1], label=noise[1])

b3 = plt.bar(index + 2 * bar_width, ade_list[2, :, 0], bar_width, alpha=0.5, color=colors[2], label=noise[2])

b4 = plt.bar(index + 3 * bar_width, ade_list[3, :, 0], bar_width, alpha=0.5, color=colors[3], label=noise[3])

# x축 위치를 정 가운데로 조정하고 x축의 텍스트를 year 정보와 매칭
plt.xticks(np.arange(bar_width, 4 + bar_width, 1), aps)

# 숫자 넣는 부분
for bar in (b1, b2, b3, b4):
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size=12)

plt.ylabel('Average Distance Error (cm)', fontsize=15)
plt.xlabel('Number of APs', fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

# print()
