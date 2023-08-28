import numpy as np
import matplotlib.pyplot as plt

import denoising_AE
import rssi_cnn
import rssi_kalman_dnn

color = ['#6EA7C1', '#6B55AE', '#E67FA2']

dae_history = denoising_AE.run_DAE(ws=20, noise_number=2, noise_f=0.5, gaus_sigma=0.5, data_num=1)
cnn_history = rssi_cnn.run_cnn(ws=20, data_num=1)
kalman_history = rssi_kalman_dnn.run_kalman_cnn(ws=20, data_num=1)

plt.figure()
# plt.plot(dae_history.history['loss'], label='loss')
plt.plot(dae_history.history['val_loss'], label='proposed model with pre-training', marker='.', color=color[0])
plt.plot(cnn_history.history['val_loss'], label='proposed model without pre-training', marker='.', color=color[1])
plt.plot(kalman_history.history['val_loss'], label='kalman filter + DNN', marker='.', color=color[2])
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.ylim([0, 1])
# plt.plot(dae_history.history['accuracy'], label='acc')
plt.plot(dae_history.history['val_accuracy'], label='proposed model with pre-training', marker='.', color=color[0])
plt.plot(cnn_history.history['val_accuracy'], label='proposed model without pre-training', marker='.', color=color[1])
plt.plot(kalman_history.history['val_accuracy'], label='kalman filter + DNN', marker='.', color=color[2])
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.show()
