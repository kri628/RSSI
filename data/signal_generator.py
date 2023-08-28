import matplotlib.pyplot as plt
import numpy as np
import random

# Sample configuration
num_samples = 10000

# Intrasample configuration
num_elements = 50
ap = 9
interval_per_element = 1
total_num_elements = int(num_elements / interval_per_element)
# starting_point = int(0 - 0.5 * total_num_elements)
starting_point = 0

# Other configuration
num_samples_visualize = 3

# Containers for samples and subsamples
samples = []
sample = []

# Generate samples
for j in range(0, num_samples):
    # Report progress
    if j % 100 == 0:
        print(j)
    # Generate wave
    rand = [random.random() for _ in range(ap)]
    # rand = random.random()
    for i in range(starting_point, total_num_elements):
        x_val = i * interval_per_element
        # y_val = x_val * x_val
        # y_val = rand

        sample.append([x_val, *rand])
        # sample.append([x_val, y_val])
    # Append wave to samples
    samples.append(sample)
    # Clear subsample containers for next sample
    sample = []

# Input shape
print(np.shape(np.array(samples[0])))
# (100, 9)

# Save data to file for re-use
np.save('./signal_waves_line_36_' + str(num_elements) + '.npy', samples)
print('saved')

# # Visualize a few random samples
# for i in range(0, num_samples_visualize):
#     random_index = np.random.randint(0, len(samples) - 1)
#     sam = np.array(samples[random_index])
#     x_axis = sam[:, 0]
#     y_axis = sam[:, 1:]
#
#     for j in range(10):
#         plt.plot(x_axis[:, j], y_axis[:, j])
#     plt.ylim([0, 1])
#     plt.title(f'Visualization of sample {random_index} ---- y: f(x) = c')
#     plt.show()
