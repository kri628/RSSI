import random

import numpy as np
import matplotlib.pyplot as plt

data = np.load('npy/dae_reconstructions_rssi_20.npz')
x_train = data['x_train']
decoded_train = data['decoded_train']

print(x_train.shape, decoded_train.shape)

sample = 178492

for j in range(10):
    index = random.randint(0, x_train.shape[0])
    # print(index)
    plt.figure(figsize=(8, 4), dpi=200)
    # plt.suptitle(str(index))
    ax = [plt.subplot(121), plt.subplot(122)]
    for i in range(8):
        ax[0].plot(x_train[index, :, i], label='AP' + str(i+1))
        ax[1].plot(decoded_train[index, :, i], label='AP' + str(i+1))

    for i in range(2):
        ax[i].set_ylim([0.2, 0.8])
        ax[i].set_xlabel('time(unit: 100 msec)', fontsize=10)
        ax[i].set_ylabel('RSSI(dB)', fontsize=10)

    ax[0].set_title('Noisy RSSI', fontsize=13)
    ax[1].set_title('Denoised RSSI', fontsize=13)

    ax[1].legend(ncol=4, fontsize=8, fancybox=False)
    plt.tight_layout()
    plt.show()
