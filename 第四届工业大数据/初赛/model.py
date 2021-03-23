# 导入包
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout,TimeDistributed
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

trainData2 = pd.read_csv('trainData2.csv', usecols=['Qi', 'Rain_sum']).values
# 验证数据为2017年的数据
validData = pd.read_csv('validData.csv', usecols=['Qi', 'Rain_sum']).values
testData1 = pd.read_csv('testData1.csv', usecols=['Qi', 'Rain_sum']).values
# print('trainData1', trainData1)
# print('trainData1', trainData1.shape)
# print('trainData2', trainData2)
# print('trainData2', trainData2.shape)
# print('validData', validData)
# print('validData', validData.shape)
# print('testData1', testData1)
# print('testData1', testData1.shape)
n_weeks = 4   # 可调
n_input = n_weeks*7*8
n_out = 7*8
n_features = 2
train_x1, train_y1 = to_supervised(trainData1, n_input, n_out, n_features)
train_x2, train_y2 = to_supervised(trainData2, n_input, n_out, n_features)
valid_x, valid_y = to_supervised(validData, n_input, n_out, n_features)
# print('train_x1', train_x1)
# print('train_y1', train_y1)
print('train_x2', train_x2)
print('train_x2', train_x2.shape)
print('train_y2', train_y2)
print('valid_x', valid_x)
print('valid_y', valid_y)
print('valid_y', valid_y.shape)
#model = Sequential()
# 设置参数
n_timesteps = 3
n_outputs = 2
epochs = 1  # 迭代次数 5次loss: loss: 0.0263 - val_loss: 0.0209
batch = 24   # 每次迭代batch数
verbose = 1  # verbose：日志显示 verbose = 0 为不在标准输出流输出日志信息 verbose = 1 为输出进度条记录 verbose = 2 为每个epoch输出一行记录
# 模型
'''model = tf.keras.Sequential([
    LSTM(1, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(100,activation='relu'),
    Dropout(0.2),
    Dense(1, activation='relu'),
])'''
model = tf.keras.Sequential([
    LSTM(3, activation='tanh', return_sequences=True, input_shape=(n_timesteps, n_features)),
    LSTM(3, activation='tanh', return_sequences=True,dropout=0.2,recurrent_dropout=0.2),
    LSTM(3, activation='tanh',dropout=0.2,recurrent_dropout=0.2),
    Dense(n_outputs)
])


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差

# 模型训练
model.fit(train_x2, train_y2, epochs=epochs, batch_size=batch, shuffle=False,
              validation_data = (valid_x, valid_y), verbose=verbose)
model.summary()
test1 = testData1[248-n_weeks*7*8:248,0:n_features].reshape((1, n_weeks*7*8, n_features))
test2 = testData1[2*248-n_weeks*7*8:2*248,0:n_features].reshape((1, n_weeks*7*8, n_features))
test3 = testData1[3*248-n_weeks*7*8:3*248,0:n_features].reshape((1, n_weeks*7*8, n_features))
#testData = np.vstack((test1,test2,test3))
print(train_x2.shape)
print(train_x2.dtype)
print(train_y2.shape)
print(test1)
print(test1.dtype)
print(test1.shape)

yhat = []
yhat = model.predict(train_x2)
print(yhat)
print(yhat.shape)
'''yhat[1] = model.predict(test2)
yhat[2] = model.predict(test3)
# 将结果写入CSV文件
submit = pd.read_csv('submission.csv',index_col=0)
for i in range(3):
    submit.iloc[i] = yhat[i]
submit.to_csv("submissiontest.csv")'''