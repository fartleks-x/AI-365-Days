# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 读取数据
data = pd.read_csv('data/diabetes.csv')

# 处理缺失值（0值替换为NaN并填充）
cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_zero] = data[cols_zero].replace(0, np.nan)
for col in cols_zero:
    data[col] = data.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))

# 划分特征和标签
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 网格搜索最佳K值（使用F1评分）
param_grid = {'n_neighbors': range(1, 31, 2)}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
grid.fit(X_train_scaled, y_train)

print("Best K:", grid.best_params_['n_neighbors'])
best_knn = grid.best_estimator_

# 预测和概率
y_pred = best_knn.predict(X_test_scaled)
y_proba = best_knn.predict_proba(X_test_scaled)[:, 1]

# 评估报告
print(classification_report(y_test, y_pred, target_names=['Non-diabetic', 'Diabetic']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-diabetic', 'Diabetic'], 
            yticklabels=['Non-diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.show()

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.plot(fpr, tpr, label=f'KNN (AUC = {auc:.2f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()