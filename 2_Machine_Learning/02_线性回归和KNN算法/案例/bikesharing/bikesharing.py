import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# 一、读取数据集
dataset = pd.read_csv('data\day.csv')

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
x, x_test, y, y_test = train_test_split(dataset[features], dataset['cnt'], test_size=0.33, random_state=42)
'''
test_size=0.33: 这表示测试集占整个数据集的33%，剩下的67%将用于训练。
random_state=42: 这是一个随机数种子，确保每次运行代码时划分数据集的方式相同，从而使结果可复现。
train_test_split() 函数会返回四个对象：
x: 训练集的特征数据。
y: 训练集的标签数据。
x_test: 测试集的特征数据。
y_test: 测试集的标签数据。
'''

print('x(训练集特征) shape is {}'.format(x.shape))
print('y(训练集标签) shape is {}'.format(y.shape))
print('-'*30)
print('x(测试集特征) shape is {}'.format(x_test.shape))
print('y(测试集标签) shape is {}'.format(y_test.shape))

# 四、训练模型：sklearn 线性回归
model_lr = linear_model.LinearRegression()
start = time.time()
model_lr.fit(x, y)
end = time.time()
print('Training time for sklearn LinearRegression: {:.4f} seconds'.format(end - start))

# 预测（sklearn 线性回归）
predictions = model_lr.predict(x_test)

# 五、梯度下降法实现（用于对比）
def normalize_features(x, mean=None, std=None):
    """标准化特征：对于梯度下降通常需要标准化才能快速收敛。"""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm, mean, std


def add_bias(x):
    """为样本矩阵增加偏置项（常数列）。"""
    return np.hstack([np.ones((x.shape[0], 1)), x])


def gradient_descent(x, y, lr=0.01, n_iter=1000):
    """简单的批量梯度下降（Batch Gradient Descent）。

    Args:
        x: 2D ndarray, 已包含偏置项
        y: 1D ndarray, 目标值
        lr: 学习率
        n_iter: 迭代次数

    Returns:
        theta: 1D ndarray, 参数向量
    """
    m, n = x.shape
    theta = np.zeros(n)
    for i in range(n_iter):
        preds = x.dot(theta)
        error = preds - y
        grad = (1 / m) * x.T.dot(error)
        theta -= lr * grad
    return theta


# 5.1 数据准备（梯度下降）
x_train_np = x.values.astype(float)
x_test_np = x_test.values.astype(float)
y_train = y.values.astype(float)
y_test_np = y_test.values.astype(float)

x_train_norm, mu, sigma = normalize_features(x_train_np)
x_test_norm = (x_test_np - mu) / (sigma + 1e-8)

x_train_b = add_bias(x_train_norm)
x_test_b = add_bias(x_test_norm)

# 5.2 训练并预测（梯度下降）
theta = gradient_descent(x_train_b, y_train, lr=0.1, n_iter=10000)
predictions_gd = x_test_b.dot(theta)

# 六、可视化结果对比
plt.style.use('fivethirtyeight')
plt.figure(figsize=(16,6))
plt.plot(y_test.values, marker='.', label='actual')
plt.plot(predictions.flatten(), marker='.', label='sklearn LinearRegression', color='r')
plt.plot(predictions_gd, marker='.', label='gradient descent', color='g')
plt.legend(loc='best')
plt.show()

# 七、计算MAE和MSE
MAE_lr = mean_absolute_error(y_test, predictions)
MSE_lr = mean_squared_error(y_test, predictions)
MAE_gd = mean_absolute_error(y_test_np, predictions_gd)
MSE_gd = mean_squared_error(y_test_np, predictions_gd)
print('MAE_lr:{0}, MSE_lr:{1}'.format(MAE_lr, MSE_lr))
print('MAE_gd:{0}, MSE_gd:{1}'.format(MAE_gd, MSE_gd))

