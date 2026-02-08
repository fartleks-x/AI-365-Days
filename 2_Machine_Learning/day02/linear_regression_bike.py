import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib


def find_target(df):
    for c in ['cnt', 'count', 'total']:
        if c in df.columns:
            return c
    raise ValueError('找不到目标列，请确保 CSV 包含 `cnt` 或 `count` 字段。')


def main():
    csv_path = 'bike-day.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 {csv_path}，请把脚本放在包含该文件的目录运行。")

    df = pd.read_csv(csv_path)
    target_col = find_target(df)

    # 丢弃可能的泄露列
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 将部分列视为分类变量
    cat_cols = [c for c in ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'] if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 填充或删除缺失值（本数据集通常没有缺失）
    X = X.fillna(X.mean())

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # 评估
    y_train_pred = model.predict(X_train_s)
    y_test_pred = model.predict(X_test_s)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print('Train RMSE: {:.3f}'.format(train_rmse))
    print('Test  RMSE: {:.3f}'.format(test_rmse))
    print('Train R2:   {:.3f}'.format(train_r2))
    print('Test  R2:   {:.3f}'.format(test_r2))

    # 保存模型与 scaler
    os.makedirs('output', exist_ok=True)
    joblib.dump(model, os.path.join('output', 'linear_model.joblib'))
    joblib.dump(scaler, os.path.join('output', 'scaler.joblib'))

    # 绘制预测 vs 实际
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual (Test)')
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'pred_vs_actual.png'))
    print('模型与图像已保存到 output/ 目录。')


if __name__ == '__main__':
    main()
