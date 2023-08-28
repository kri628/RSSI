import math

import matplotlib.pyplot as plt
import numpy as np

algorithm = {'DAE with\npre-training': [0.7354282304562741, 1.254819933322424, 53.85412822437606/2.5],
             'without\npre-training': [0.64, 1.71, 55.52687688790598/2],
             'Kalman Filter': [0.62, 1.77, 58.91210307090989/2],
             'Trilateration': [0.207598273241974744, 7.539780387150145, 120.78038150831057],
             'Kalman Filter\n+\nTrilateration': [0.3143460036542113, 5.666026708350406, 66.52985078582702],
             'DAE\n+\nTrilateration': [0.4123507006084004, 5.637922682870927, 102.78038150831057]
             }
# acc = [0.7354282304562741, 0.64, 0.62, 0.207598273241974744, 0.3143460036542113]
# ade = np.array([1.254819933322424, 1.71, 1.77, 5.666026708350406, 5.637922682870927]) * 20
color = ['#a8e8f9', '#00537a', '#013c58', '#f5a201', '#ffba42', '#ffd35b']
x = np.arange(len(algorithm))

values = np.array(list(algorithm.values()))
print(values)

plt.figure(figsize=(8, 5), dpi=200)
# ax0 = plt.subplot(121)
# ax0.bar(x[:3], values[:3, 0], color=color)
# ax0.set_xticks(x[:3], list(algorithm.keys())[:3])
# ax0.set_ylabel('Classification Accuracy', fontsize=13)
# ax0.set_ylim([0, 1])

y = values[:, 1] * 20
ax1 = plt.subplot(111)
ax1.bar(x, y, yerr=values[:, 2], capsize=10, color=color)
ax1.set_xticks(x, algorithm.keys())
# ax1.set_ylim([-50, 200])
ax1.set_ylabel('Average Distance Error (cm)', fontsize=13)

for i, v in enumerate(x):
    plt.text(v, y[i]+values[i, 2], np.around(y[i], 2),                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize=9,
             # color='blue',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')

plt.tight_layout()
plt.show()

print(values[:, 1] * 20)
