# 导入包
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten,GRU,RNN
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
trainData1 = pd.read_csv('trainData1.csv', usecols=['Qi']).values
print(trainData1)

print(trainData1.shape)
print(trainData1.dtype)
trainData2 = pd.read_csv('trainData2.csv', usecols=['Qi']).values
# 验证数据为2017年的数据
validData = pd.read_csv('validData.csv', usecols=['Qi']).values
testData1 = pd.read_csv('C:\\Users\\d1526\\Desktop\\比赛\\工业大数据比赛\\决赛\\testData1.csv', usecols=['Qi', 'Rain_sum']).values
n_weeks = 3   # 可调
n_input = n_weeks*7*8
n_out = 7*8
n_features = 1
train_x1, train_y1 = to_supervised(trainData1, n_input, n_out, n_features)
train_x2, train_y2 = to_supervised(trainData2, n_input, n_out, n_features)
valid_x, valid_y = to_supervised(validData, n_input, n_out, n_features)
model = Sequential()
# 设置参数
n_timesteps = 168
n_outputs = 56
epochs = 5  # 迭代次数 5次loss: loss: 0.0263 - val_loss: 0.0209
batch = 24   # 每次迭代batch数
verbose = 1  # verbose：日志显示 verbose = 0 为不在标准输出流输出日志信息 verbose = 1 为输出进度条记录 verbose = 2 为每个epoch输出一行记录
print(train_x1.shape)
print(valid_x.shape)
# 模型一 0.3 loss: 0.0377 - val_loss: 0.0320
'''model.add(LSTM(1, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(1, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='relu')))
model.add(TimeDistributed(Dense(1)))'''
# 模型二 loss: 0.0235 - val_loss: 0.0197 引用
'''model.add(LSTM(3, activation='tanh', return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(LSTM(3, activation='tanh', return_sequences=True,dropout=0.2,recurrent_dropout=0.2))
model.add(LSTM(3, activation='tanh',dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(n_outputs))'''
# 模型三
# 3 loss: 0.0225 - val_loss: 0.0211
# 2 loss: 0.0292 - val_loss: 0.0269
# 4 loss: 0.0209 - val_loss: 0.0206
# 5 loss: 0.0206 - val_loss: 0.0197
# 6 loss: 0.0204 - val_loss: 0.0204
# 7 loss: 0.0212 - val_loss: 0.0210
model.add(Bidirectional(LSTM(5, activation='relu'), input_shape=(n_timesteps, n_features)))
model.add(Dense(n_outputs))
# 模型四 loss: 0.0228 - val_loss: 0.0268
'''model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(n_outputs))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(5, activation='relu')))
model.add(TimeDistributed(Dense(1)))'''

'''model.add(GRU(3, activation='tanh', return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(GRU(3, activation='tanh', return_sequences=True,dropout=0.2,recurrent_dropout=0.2))
model.add(GRU(3, activation='tanh',dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(n_outputs))'''


'''model.add(Bidirectional(GRU(5, activation='relu'), input_shape=(n_timesteps, n_features)))
model.add(Dense(n_outputs))'''




# 模型编译
model.compile(optimizer='adam', loss='mse')


# 模型训练
model.fit(train_x2, train_y2, epochs=epochs, batch_size=batch, shuffle=False,
              validation_data = (valid_x, valid_y), verbose=verbose)
# callbacks=[early_stop]
model.summary()

#测试集
valid_y1 = model.predict(valid_x)
print(valid_y1)


from matplotlib import pyplot as plt
plt.figure(figsize=(20,6))

plt.plot(valid_y,color='red')
plt.plot(valid_y1,color='green')
plt.legend()  # 画出曲线图标
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

##########evaluate##############
#均值
mean = np.mean(valid_y1)
#标准差
std = np.std(valid_y1)
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(valid_y, valid_y1)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(valid_y, valid_y1))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(valid_y, valid_y1)
print("平均值为：%f" % mean)
print("标准差为:%f" % std)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)














'''test1 = testData1[248-n_weeks*7*8:248,0:n_features].reshape((1, n_weeks*7*8, n_features))
test2 = testData1[2*248-n_weeks*7*8:2*248,0:n_features].reshape((1, n_weeks*7*8, n_features))
test3 = testData1[3*248-n_weeks*7*8:3*248,0:n_features].reshape((1, n_weeks*7*8, n_features))
test4 = testData1[4*248-n_weeks*7*8:4*248,0:n_features].reshape((1, n_weeks*7*8, n_features))
test5 = testData1[5*248-n_weeks*7*8:5*248,0:n_features].reshape((1, n_weeks*7*8, n_features))
testData = np.vstack((test1, test2, test3,test4,test5))

yhat = model.predict(testData).reshape((5, 56))
# 将结果写入CSV文件
submit = pd.read_csv('submission.csv', index_col=0)
for i in range(len(yhat)):
    submit.iloc[i] = yhat[i]
submit.to_csv("submissionBLSTM1.csv")'''