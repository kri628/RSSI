import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-notebook')

# Sample configuration
num_samples_visualize = 10
noise_factor = 0.1

ws = 30
noise_name = ['Gaussian', 'Proportional factor', 'Occurrence Probability', 'Standard deviation']

# linestyle = ['solid', 'dashed', 'dotted', 'dashdot']
# color = ['black', 'gray']

# # Load data
# data = np.load('./signal_waves_line_' + str(ws) + '.npy')   # (100000, 20, 5)
# x_val = data[:, :, 0]
# y_val = data[:, :, 1:]
# print(x_val.shape, y_val.shape)
# # (100000, 20) (100000, 20, 8)
#
# # Add noise to data
# noisy_samples = []
# for i in range(0, len(x_val)):
#     if i % 100 == 0:
#         print(i)
#     pure = np.array(y_val[i])
#     noise2 = np.random.normal(0, 0.5, pure.shape)
#     noise1 = np.random.binomial(1, 0.3, pure.shape)
#     signal = pure + noise_factor * noise1 * noise2
#     sample = np.concatenate((np.array(x_val[i]).reshape((ws, 1)), signal), axis=1)
#     noisy_samples.append(sample)
#
# # Save data to file for re-use
# # np.save('./signal_waves_noisy_line_' + str(ws) + '.npy', noisy_samples)
# # (100000, 100, 5)


def noise(pure, noise_number, noise_f=0.01, gaus_sigma=0.5):
    if noise_number == 0:
        noise = np.random.normal(0, gaus_sigma, pure.shape)
        # noise_factor = noise_f
        # noise1 = np.random.binomial(1, 0.3, pure.shape)
        # signal = pure + noise_f * noise
        return pure * (1 - noise_f) + noise_f * noise

    elif noise_number == 1:
        # noise 1
        noise = np.random.normal(0, gaus_sigma, pure.shape)
        # k = noise_f
        noise_factor = (1 - pure) * noise_f + 0.0001
        # signal = pure + noise_factor * noise
        return pure * (1 - noise_f) + noise_factor * noise

    elif noise_number == 2:
        # noise 2
        noise = np.random.normal(0, gaus_sigma, pure.shape)
        # noise_factor = 0.05
        rand = np.random.rand(pure.shape[0], pure.shape[1])
        p = pure
        f = np.where(rand >= p, 1, 0)
        # signal = pure + noise * f * noise_f
        return pure * (1 - noise_f) + noise * f * noise_f

    elif noise_number == 3:
        # noise 3
        sigma = 1 - pure
        noise = np.random.normal(0, sigma, pure.shape)
        # noise_factor = 0.01
        # signal = pure + noise * noise_f
        return pure * (1 - noise_f) + noise * noise_f

    else:
        print("noise number error")
        return


# Load data
data = np.load('./signal_waves_line_' + str(ws) + '.npy')  # (100000, 20, 5)
x_val_pure = data[:, :, 0]
y_val_pure = data[:, :, 1:]
# print(x_val_pure.shape, y_val_pure.shape)
# (100000, 20) (100000, 20, 8)

def generate_samples(noise_number):
    # Add noise to data
    noisy_samples = []
    for i in range(0, len(x_val_pure)):

        pure = np.array(y_val_pure[i])

        signal = noise(pure, noise_number=noise_number, noise_f=0.1)

        sample = np.concatenate((np.array(x_val_pure[i]).reshape((ws, 1)), signal), axis=1)
        noisy_samples.append(sample)

    return np.array(noisy_samples)
    # x_val_noisy, y_val_noisy = noisy_samples[:, :, 0], noisy_samples[:, :, 1:]


noisy_samples = [generate_samples(0), generate_samples(1), generate_samples(2), generate_samples(3)]


# Visualize a few random samples
for i in range(num_samples_visualize):
    random_index = np.random.randint(0, len(noisy_samples[0]) - 1)

    # origin_sam = y_val_pure[random_index]

    sam = noisy_samples[0][random_index]
    x_axis = sam[:, 0]
    y_axis = sam[:, 1:]

    plt.figure(figsize=(8, 8), dpi=200)
    # ax0 = plt.subplot(141)
    # ax0.set_ylim([0, 1])
    ax = [plt.subplot(221),
          plt.subplot(222),
          plt.subplot(223),
          plt.subplot(224)]

    for i in range(4):
        ax[i].set_ylim([0, 1])
        ax[i].set_xlabel('time(unit: 100 msec)')
        ax[i].set_ylabel('RSSI(dB)')
        ax[i].set_title(noise_name[i])

        sam = noisy_samples[i][random_index]
        x_axis = sam[:, 0]
        y_axis = sam[:, 1:]
        for j in range(8):
            # print(color[j//4], linestyle[j%4])
            # ax0.plot(x_axis, origin_sam[:, j], label='AP' + str(j+1))
            ax[i].plot(x_axis, y_axis[:, j], label='AP' + str(j+1))


    # ax0.set_title('Pure sample', fontsize=20)
    # ax1.set_title('Noisy sample', fontsize=20)
    ax[1].legend(loc='lower left', mode='expand', ncol=4, bbox_to_anchor=(0, 1.1, 1, 0.2), fontsize=8, fancybox=False)

    plt.tight_layout()
    plt.show()

