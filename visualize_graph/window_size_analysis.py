import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rssi_estimate

colors = ['red', 'green', 'blue', 'darkorchid']
noise = ['Gaussian', 'Proportional factor', 'Occurrence Probability', 'Standard deviation']
window_size = (10, 20, 30, 40, 50)
marker = ['.', '^', 'p', '*']

acc_arr, ade_arr = [], []
for i in range(4):
    acc_inner, ade_inner = [], []
    for ws in window_size:
        df = pd.read_csv(f'predict/dae2/220920_noise{i}_03_{ws}.csv')
        # print(i, ws)
        acc, ade, ade_cm = rssi_estimate.estimate(df, False)
        acc_inner.append(acc)
        ade_inner.append(ade_cm)
    acc_arr.append(acc_inner)
    ade_arr.append(ade_inner)

print(acc_arr)

x = np.arange(len(window_size))
plt.figure(figsize=(6, 4), dpi=200)
plt.ylim([0.5, 0.8])
for i in range(4):
    plt.plot(acc_arr[i], label=noise[i], color=colors[i], marker=marker[i])
plt.xticks(x, labels=window_size)
plt.ylabel('Classification Accuracy', fontsize=12)
plt.xlabel('Window Length', fontsize=12)
plt.legend(fontsize=10, fancybox=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4), dpi=200)
plt.ylim([10, 50])
for i in range(4):
    plt.plot(ade_arr[i], label=noise[i], color=colors[i], marker=marker[i])
plt.xticks(x, labels=window_size)
plt.ylabel('Average Distance Error (cm)', fontsize=12)
plt.xlabel('Window Length', fontsize=12)
plt.legend(fontsize=10, fancybox=False)
plt.tight_layout()
plt.show()
