import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 一、读取数据集
dataset = pd.read_csv('day.csv')

# 二、数据预处理
dataset = dataset.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)
'''
axis=1: 这是关键参数，它告诉 .drop() 方法操作的对象是列。
axis=0 表示删除行（默认值）。
axis=1 表示删除列。
'''
print(dataset.info())

features = list(dataset.columns.values) # 获取数据集的列名，并将其转换为列表
print("Features:", features)

features.remove('cnt')
print("Features after removing 'cnt':", features)

# 三、划分训练集和测试集
X, X_test, y, y_test = train_test_split(dataset[features], dataset['cnt'], test_size=0.2, random_state=42)

print('X(训练集特征) shape is {}'.format(X.shape))
print('y(训练集标签) shape is {}'.format(y.shape))
print('-'*30)
print('X(测试集特征) shape is {}'.format(X_test.shape))
print('y(测试集标签) shape is {}'.format(y_test.shape))

# 四、训练模型
model_lr = linear_model.LinearRegression() 
model_lr.fit(X, y) 

# 五、模型评估
predictions = model_lr.predict(X_test)
print(predictions.shape)
print(predictions.flatten())
print(y_test.values)

# 六、可视化结果
plt.style.use('fivethirtyeight')
plt.figure(figsize=(16,6))
plt.plot(y_test.values, marker=".", label="actual")
plt.plot(predictions.flatten(), marker=".", label="predicttion", color="r")
plt.legend(loc="best")
plt.show()

# 七、计算MAE和MSE
MAE_lr = mean_absolute_error(y_test, predictions)
MSE_lr = mean_squared_error(y_test, predictions)
print('MAE_lr:{0}, MSE_lr:{1}'.format(MAE_lr, MSE_lr))

