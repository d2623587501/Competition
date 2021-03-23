from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
##################################
# 读取数据

# 读取入库流量数据
dataA = pd.read_excel('入库流量数据.xlsx')
#print(dataA)
dataA = np.array(dataA['Qi'])

#print(dataA)
dataA1 = dataA[0:2920]
dataA2 = dataA[2920:5507]
dataA3 = dataA[5507:8427]
dataA4 = dataA[8427:11356]
dataA5 = dataA[11356:14275]


plt.figure(figsize=(20,6))

plt.subplot(231)
plt.title('2013')
plt.plot(dataA1)
plt.legend()  # 画出曲线图标

plt.subplot(232)
plt.title('2014')
plt.plot(dataA2)
plt.legend()  # 画出曲线图标

plt.subplot(233)
plt.title('2015')
plt.plot(dataA3)
plt.legend()  # 画出曲线图标

plt.subplot(234)
plt.title('2016')
plt.plot(dataA4)
plt.legend()  # 画出曲线图标

plt.subplot(235)
plt.title('2017')
plt.plot(dataA5)
plt.legend()  # 画出曲线图标
plt.show()