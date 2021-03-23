# 导入包
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
import numpy as np
import os
import tensorflow as tf
# 用之前的数据去预测未来的值，可以滑动构造数据
# train为数据，多维的时间序列；n_input为输入的维数，n_out为预测的步数，n_features为要使用前几个时间序列
def to_supervised(train, n_input, n_out, n_features):
    data = train
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, 0:n_features])  # 使用几个特征
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)
# 最多可以用一个月（4周）的数据预测未来一周的流量
# trainData1.csv 13-14年、trainData2.csv 15-16年、validData.csv 17年、testData1.csv 需要预测的3个时间段
# baseline中只用了2个feature
trainData1 = pd.read_csv('trainData1.csv', usecols=['Qi', 'Rain_sum']).values
print(trainData1)
print(trainData1.shape)
print(trainData1.dtype)
trainData2 = pd.read_csv('trainData2.csv', usecols=['Qi', 'Rain_sum']).values
# 验证数据为2017年的数据
validData = pd.read_csv('validData.csv', usecols=['Qi', 'Rain_sum']).values
testData1 = pd.read_csv('C:\\Users\\d1526\\Desktop\\比赛\\工业大数据比赛\\初赛\\testData1.csv', usecols=['Qi', 'Rain_sum']).values
n_weeks = 3   # 可调
n_input = n_weeks*7*8
n_out = 7*8
n_features = 2
train_x1, train_y1 = to_supervised(trainData1, n_input, n_out, n_features)
train_x2, train_y2 = to_supervised(trainData2, n_input, n_out, n_features)
valid_x, valid_y = to_supervised(validData, n_input, n_out, n_features)
print(train_x2.shape)
print(train_y2.shape)
print(valid_x.shape)
print(valid_y.shape)