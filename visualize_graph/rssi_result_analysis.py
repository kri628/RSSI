import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import rssi_estimate

# filelist = glob.glob('predict/dae_predict/220920_*.csv')

noise_factor = (
                # '0.01', '0.03', '0.05',
                '0.1', '0.2', '0.3', '0.4', '0.5')
colors = ['red', 'green', 'blue', 'darkorchid']
noise = ['Gaussian', 'Proportional factor', 'Occurrence Probability', 'Standard deviation']
marker = ['.', '^', 'p', '*']

filelist, acc_list, ade_list = [], [], []
for noise_number in range(4):
    name_arr = []
    acc_arr = []
    ade_arr = []
    for noise_f in (
                    # '001', '003', '005',
                    '01', '02', '03', '04', '05'):
        name_inner = sorted(glob.glob(f'predict/dae_predict/220920_noise{noise_number}_{noise_f}_20.csv'))
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

print(np.array(filelist))

filelist = np.array(filelist)
acc_list = np.array(acc_list)
ade_list = np.array(ade_list)
print(filelist, acc_list, ade_list, sep='\n')

x = np.arange(len(noise_factor))
figsize = (6, 4)

plt.figure(figsize=figsize, dpi=200)
plt.ylim([0.6, 0.75])
for i in range(4):
    plt.plot(acc_list[i, :, 0], label=noise[i], color=colors[i], marker=marker[i])
# plt.title('Accuracy(window size: 20)', fontsize=13)
plt.xticks(x, labels=noise_factor)
plt.ylabel('Classification Accuracy', fontsize=12)
plt.xlabel('noise factor', fontsize=12)
plt.legend(fontsize=10, fancybox=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=figsize, dpi=200)
plt.ylim([20, 40])
for i in range(4):
    plt.plot(ade_list[i, :, 0], label=noise[i], color=colors[i], marker=marker[i])
# plt.title('Average Distance Error(window size: 20)', fontsize=13)
plt.xticks(x, labels=noise_factor)
plt.ylabel('Average Distance Error (cm)', fontsize=12)
plt.xlabel('noise factor', fontsize=12)
plt.legend(fontsize=10, fancybox=False)
plt.tight_layout()
plt.show()

acc_list = np.array(acc_list)
print(acc_list[2] - acc_list[0])

# plt.figure(figsize=figsize, dpi=200)
# plt.ylim([0.6, 0.75])
# for i in range(4):
#     plt.plot(acc_list[i, :, 1], label=noise[i], color=colors[i])
# # plt.title('Accuracy', fontsize=13)
# plt.xticks(x, labels=noise_factor)
# plt.ylabel('Classification Accuracy', fontsize=10)
# plt.xlabel('noise factor', fontsize=10)
# plt.legend(fontsize=10, fancybox=False)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=figsize)
# plt.ylim([20, 40])
# for i in range(4):
#     plt.plot(ade_list[i, :, 1], label=noise[i], color=colors[i])
# # plt.title('Average Distance Error (cm)', fontsize=13)
# plt.xticks(x, labels=noise_factor)
# plt.ylabel('Average Distance Error (cm)', fontsize=10)
# plt.xlabel('noise factor', fontsize=10)
# plt.legend(fontsize=10, fancybox=False)
# plt.tight_layout()
# plt.show()
