# 导入包
import numpy as np
import pandas as pd
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
# 读入数据
testData1 = pd.read_csv('testData1.csv', usecols=['Qi', 'Rain_sum','T','w','wd']).values
print(testData1)
print(testData1.shape)
n_weeks = 2   # 可调
n_input = n_weeks*7*8
n_out = 7*8
n_features = 2
test_x, test_y = to_supervised(testData1, n_input, n_out, n_features)
print(test_x)
print(test_x.shape)
print(test_y)
print(test_y.shape)