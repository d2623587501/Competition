# 导入包
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

##################################
# 读取数据

# 读取入库流量数据
data = pd.read_excel('入库流量数据.xlsx')
Qi_train1 = data[0:5507].copy().reset_index(drop=True)  # 第一段数据
Qi_train2 = data[5507:14275].copy().reset_index(drop=True)  # 第二段数据
Qi_test1 = data[14275:].copy().reset_index(drop=True)  # 测试数据
# print(Qi_test1)
# print(Qi_train1)
# print(Qi_train2)
# 读取遥测站降雨数据
Raindata = pd.read_excel('遥测站降雨数据.xlsx')
raindata_new = pd.DataFrame()
raindata_new['TimeStample'] = Raindata['TimeStample'].copy()
Raindata.drop('TimeStample', axis=1, inplace=True)
# 将所有测点的数据求和作为总降雨
raindata_new['Rain_sum'] = Raindata.apply(lambda x: x.sum(), axis=1)
Rain_train = raindata_new[0:43824].copy().reset_index(drop=True)
Rain_test1 = raindata_new[43824:].copy().reset_index(drop=True)
# print(Rain_train)
# print(Rain_test1)
# 对降雨数据进行重采样
Rain_train.set_index('TimeStample', inplace=True)
# 流量的采样频率为（每3个小时），而降雨为每1个小时
Rain_train = Rain_train.resample('3H').sum()
Qi_train1['Rain_sum'] = Rain_train['Rain_sum'].values[0:5507]
Qi_train2['Rain_sum'] = Rain_train['Rain_sum'].values[5507+333:]
Rain_test1.set_index('TimeStample', inplace=True)
a1 = Rain_test1[0:31*24].resample('3H').sum()
a2 = Rain_test1[31*24:2*31*24].resample('3H').sum()
a3 = Rain_test1[2*31*24:3*31*24].resample('3H').sum()
Qi_test1['Rain_sum'] = pd.concat([a1, a2, a3], axis=0, ignore_index=True)
# print(Qi_train1)
# print(Qi_train2)
# print(Qi_test1)
# 读取降雨预报数据，对于缺失数据用前一天的进行补充
Environmentdata = pd.read_excel('环境表.xlsx')
# 填充缺失
Environmentdata['T'].fillna(method='ffill', inplace=True)
Environmentdata['w'].fillna(method='ffill', inplace=True)
# wd需要归一化
ss = StandardScaler()
Environmentdata['wd'] = ss.fit_transform(Environmentdata['wd'].values.reshape(-1, 1))
# 对于环境变量的采样频率为1天，所以也需要进行重采样
Environment = pd.DataFrame()
Environment['T'] = np.zeros(len(Environmentdata)*8)
Environment['w'] = np.zeros(len(Environmentdata)*8)
Environment['wd'] = np.zeros(len(Environmentdata)*8)
for i in range(len(Environment)):
    Environment['T'][i] = Environmentdata['T'][int(i/8)]
    Environment['w'][i] = Environmentdata['w'][int(i/8)]
    Environment['wd'][i] = Environmentdata['wd'][int(i/8)]
# 然后分配给两段训练数据和测试数据
Qi_train1['T'] = Environment['T'][0:len(Qi_train1)].values
Qi_train1['w'] = Environment['w'][0:len(Qi_train1)].values
Qi_train1['wd'] = Environment['wd'][0:len(Qi_train1)].values
Qi_train2['T']= Environment['T'][5840:(5840+len(Qi_train2))].values
Qi_train2['w']= Environment['w'][5840:(5840+len(Qi_train2))].values
Qi_train2['wd']= Environment['wd'][5840:(5840+len(Qi_train2))].values
Qi_test1['T']= Environment['T'][(5840+len(Qi_train2)):(5840+len(Qi_train2)+len(Qi_test1))].values
Qi_test1['w']= Environment['w'][(5840+len(Qi_train2)):(5840+len(Qi_train2)+len(Qi_test1))].values
Qi_test1['wd']= Environment['wd'][(5840+len(Qi_train2)):(5840+len(Qi_train2)+len(Qi_test1))].values
# print(Qi_train1)
# print(Qi_train2)
# print(Qi_test1)
# 数据集处理后的存储 trainData1.csv 13-14年、trainData2.csv 15-16年、validData.csv 17年、testData1.csv 需要预测的3个时间段
trainData1 = Qi_train1
trainData1.to_csv("trainData1.csv", index=False)
print(trainData1)

trainData2 = Qi_train2[:5848]
trainData2.to_csv("trainData2.csv", index=False)
print(trainData2)

validData = Qi_train2[5848:]
validData.to_csv("validData.csv", index=False)
print(validData)

testData1 = Qi_test1
testData1.to_csv("testData1.csv", index=False)
print(testData1)

