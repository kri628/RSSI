import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KalmanFilter():
    def __init__(self, process_noise=0.005, measurement_noise=20):
        super(KalmanFilter, self).__init__()
        self.isInitialized = False
        self.processNoise = process_noise
        self.measurementNoise = measurement_noise
        self.predictedRSSI = 0
        self.errorCovariance = 0

    def filtering(self, rssi):
        if not self.isInitialized:
            self.isInitialized = True
            priorRSSI = rssi
            priorErrorCovariance = 1
        else:
            priorRSSI = self.predictedRSSI
            priorErrorCovariance = self.errorCovariance + self.processNoise

        kalmanGain = priorErrorCovariance / (priorErrorCovariance + self.measurementNoise)
        self.predictedRSSI = priorRSSI + (kalmanGain * (rssi - priorRSSI))
        self.errorCovarianceRSSI = (1 - kalmanGain) * priorErrorCovariance

        return self.predictedRSSI


def exe_kalman(data, ws):

    pred_data = []
    for k in range(len(data)):
        pred_ap = []
        for j in range(8):
            kalman = KalmanFilter()
            pred_rssi = []
            for l in range(ws):
                pred_rssi.append(kalman.filtering(data[k][l][j]))
            # print(pred_rssi)
            pred_ap.append(pred_rssi)
            # print(j)
            # print(pred_ap)
        pred_ap = np.array(pred_ap).T

        pred_data.append(pred_ap)
    return pred_data




if __name__ == '__main__':

    filepath = './dataset/rssi_sum_220105.csv'

    df = pd.read_csv(filepath)
    print(df.head())

    rssi_df = df[['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8']]
    label_df = df[['x', 'y']]

    rssi = rssi_df.to_numpy()
    label = label_df.to_numpy()

    kalman = KalmanFilter()

    pred_rssi = []
    for i in range(len(rssi)):
        pred_rssi.append(kalman.filtering(rssi[i][0]))

    x = np.arange(len(rssi))
    plt.figure()
    plt.plot(x, rssi[:, 0])
    plt.plot(x, pred_rssi)
    plt.show()
