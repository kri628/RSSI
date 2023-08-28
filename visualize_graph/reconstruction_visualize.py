import numpy as np
import matplotlib.pyplot as plt
import random

plt.style.use('seaborn-v0_8-notebook')

pure, noisy, reconstruction = np.load('npy/dae_reconstructions20.npy')

print(pure.shape, noisy.shape, reconstruction.shape)

for i in range(10):
    index = random.randint(0, 30000)
    # plt.figure(figsize=(12, 4))
    plt.figure(figsize=(8, 4), dpi=200)

    ax = [plt.subplot(131), plt.subplot(132), plt.subplot(133)]

    for j in range(8):
        ax[0].plot(pure[index, :, j])
        ax[1].plot(noisy[index, :, j])
        ax[2].plot(reconstruction[index, :, j])

    for ax0 in ax:
        ax0.set_ylim([0, 1])
        ax0.set_xlabel('time(unit: 100msec)', fontsize=13)
        ax0.set_ylabel('RSSI(dB)', fontsize=13)

    plt.tight_layout()
    plt.show()

