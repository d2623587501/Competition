# 导入包
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

##################################
# 读取数据

# 读取入库流量数据
data = pd.read_excel('入库流量数据.xlsx')
Qi_test1 = data[0:].copy().reset_index(drop=True)  # 测试数据
# print(Qi_test1)
# 读取遥测站降雨数据
Raindata = pd.read_excel('遥测站降雨数据.xlsx')
raindata_new = pd.DataFrame()
raindata_new['TimeStample'] = Raindata['TimeStample'].copy()
Raindata.drop('TimeStample', axis=1, inplace=True)
# 将所有测点的数据求和作为总降雨
raindata_new['Rain_sum'] = Raindata.apply(lambda x: x.sum(), axis=1)
Rain_test1 = raindata_new[0:].copy().reset_index(drop=True)
# print(Rain_test1)
Rain_test1.set_index('TimeStample', inplace=True)
a1 = Rain_test1[0:31*24].resample('3H').sum()
a2 = Rain_test1[31*24:2*31*24].resample('3H').sum()
a3 = Rain_test1[2*31*24:3*31*24].resample('3H').sum()
a4 = Rain_test1[3*31*24:4*31*24].resample('3H').sum()
a5 = Rain_test1[4*31*24:5*31*24].resample('3H').sum()
Qi_test1['Rain_sum'] = pd.concat([a1, a2, a3, a4, a5], axis=0, ignore_index=True)
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
Qi_test1['T']= Environment['T'][(0):(len(Qi_test1))].values
Qi_test1['w']= Environment['w'][(0):(len(Qi_test1))].values
Qi_test1['wd']= Environment['wd'][(0):(len(Qi_test1))].values
# print(Qi_test1)
# 数据集保存 testData1.csv
testData1 = Qi_test1
testData1.to_csv("testData1.csv", index=False)
print(testData1)