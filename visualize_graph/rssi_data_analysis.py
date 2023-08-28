import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

df = pd.read_csv('dataset/rssi_sum_220403.csv')
print(df.head())


def distance(p1, p2):
    return int(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * 20)


x1, y1 = 0, 1
close = df[(df['x'] == x1) & (df['y'] == y1)]['rssi3'].to_numpy()
# print(close)

x2, y2 = 7, 4
mid = df[(df['x'] == x2) & (df['y'] == y2)]['rssi3'].to_numpy()
# print(mid)

# x3, y3 = 14, 9
# far = df[(df['x'] == x3) & (df['y'] == y3)]['rssi2'].to_numpy()
x3, y3 = 12, 7
far = df[(df['x'] == x3) & (df['y'] == y3)]['rssi3'].to_numpy()
# print(far)

ylim = [-80, -30]
plt.figure(figsize=(12, 4), dpi=200)
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

ax1.set_ylim([-52.5, -37.5])
ax2.set_ylim([-62.5, -47.5])
ax3.set_ylim([-80, -65])

x = np.arange(200)
ax1.plot(x, close[:len(x)])
ax2.plot(x, mid[:len(x)])
ax3.plot(x, far[:len(x)])

ax1.set_xlabel('time(unit: 100 msec)', fontsize=13)
ax1.set_ylabel('RSSI(dB)', fontsize=13)
ax2.set_xlabel('time(unit: 100 msec)', fontsize=13)
ax2.set_ylabel('RSSI(dB)', fontsize=13)
ax3.set_xlabel('time(unit: 100 msec)', fontsize=13)
ax3.set_ylabel('RSSI(dB)', fontsize=13)

ax1.set_title('Short distance (20 cm)', fontsize=13)
ax2.set_title(f'Intermediate distance ({distance((7, 4), (0, 0))} cm)', fontsize=13)
ax3.set_title(f'Long distance ({distance((12, 7), (0, 0))} cm)', fontsize=13)

plt.tight_layout()
plt.show()
