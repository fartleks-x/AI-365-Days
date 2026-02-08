"""
Pima Indians 糖尿病数据集 - KNN 分类分析
任务5.1: 下载数据集，使用KNN建模，输出预测准确率
任务5.2: 调整K值和训练/测试集配比，观察准确率变化
任务5.3: 分类问题的评判指标详解
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 任务5.1: 下载数据集并进行KNN分析 ====================

def task_5_1_basic_knn():
    """
    任务5.1: 下载Pima Indians糖尿病数据集，使用KNN进行分类建模
    """
    print("=" * 80)
    print("任务 5.1: KNN基础建模与准确率计算")
    print("=" * 80)
    
    # 数据集来源：https://www.kaggle.com/uciml/pima-indians-diabetes-database
    # 也可以直接从UCI Machine Learning Repository下载
    # 这里使用sklearn自带的方式或从网络加载
    
    # 方法一：直接下载CSV文件
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        # 尝试从网络加载
        df = pd.read_csv(url, header=None, names=columns)
        print(f"✓ 成功从网络加载数据集")
    except:
        # 如果网络不可用，创建示例数据（实际应用中应下载真实数据）
        print("⚠ 网络加载失败，使用本地示例数据进行演示")
        df = create_sample_data()
    
    print(f"\n数据集形状: {df.shape}")
    print(f"特征列: {columns[:-1]}")
    print(f"目标列: {columns[-1]}")
    print("\n数据集前5行:")
    print(df.head())
    print(f"\n数据集统计信息:\n{df.describe()}")
    print(f"\n目标值分布:\n{df['Outcome'].value_counts()}")
    
    # 数据预处理
    X = df.iloc[:, :-1].values  # 特征
    y = df.iloc[:, -1].values     # 目标标签
    
    # 标准化特征（KNN对特征尺度敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 创建KNN模型 (K=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'─' * 40}")
    print(f"基础模型性能 (K=5, test_size=0.2)")
    print(f"{'─' * 40}")
    print(f"预测准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return df, X_scaled, y


def create_sample_data():
    """创建示例数据（用于演示，实际应使用真实数据）"""
    np.random.seed(42)
    n_samples = 768
    
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(44, 200, n_samples),
        'BloodPressure': np.random.randint(24, 122, n_samples),
        'SkinThickness': np.random.randint(7, 99, n_samples),
        'Insulin': np.random.randint(14, 846, n_samples),
        'BMI': np.random.uniform(18, 67, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)


# ==================== 任务5.2: 调整K值和训练/测试集配比 ====================

def task_5_2_hyperparameter_tuning(X_scaled, y):
    """
    任务5.2: 调整不同的K值和训练/测试集配比，观察准确率变化
    """
    print("\n" + "=" * 80)
    print("任务 5.2: 超参数调整与准确率对比")
    print("=" * 80)
    
    # 测试不同的K值
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    
    # 测试不同的训练/测试集配比
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 1. 固定测试集比例(0.2)，改变K值
    print("\n【对比1】: 固定 test_size=0.2，改变 K 值")
    print("─" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    k_results = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        k_results.append({'K': k, 'Accuracy': accuracy})
        print(f"K={k:2d}  →  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. 固定K值(5)，改变测试集比例
    print("\n【对比2】: 固定 K=5，改变 test_size 比例")
    print("─" * 60)
    
    test_size_results = []
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_size = 1 - test_size
        test_size_results.append({
            'train_size': train_size,
            'test_size': test_size,
            'Accuracy': accuracy
        })
        print(f"训练集:{train_size*100:3.0f}% / 测试集:{test_size*100:3.0f}%  →  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 3. 2D热力图：K值 vs 测试集比例
    print("\n【对比3】: K值 vs 测试集比例 - 准确率热力图")
    print("─" * 60)
    
    accuracy_matrix = np.zeros((len(k_values), len(test_sizes)))
    
    for i, k in enumerate(k_values):
        for j, test_size in enumerate(test_sizes):
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_matrix[i, j] = accuracy
    
    # 打印热力图
    df_heatmap = pd.DataFrame(
        accuracy_matrix,
        index=[f'K={k}' for k in k_values],
        columns=[f'test_size={ts}' for ts in test_sizes]
    )
    print(df_heatmap.round(4))
    
    # 可视化结果
    plot_hyperparameter_results(k_values, k_results, test_size_results, 
                                accuracy_matrix, test_sizes)
    
    return k_results, test_size_results


def plot_hyperparameter_results(k_values, k_results, test_size_results, 
                               accuracy_matrix, test_sizes):
    """绘制超参数调整的可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KNN超参数调整 - 准确率对比', fontsize=16, fontweight='bold')
    
    # 子图1: K值影响
    ax1 = axes[0, 0]
    k_list = [r['K'] for r in k_results]
    accuracy_list = [r['Accuracy'] for r in k_results]
    ax1.plot(k_list, accuracy_list, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('K 值', fontsize=11)
    ax1.set_ylabel('准确率', fontsize=11)
    ax1.set_title('K值对准确率的影响 (test_size=0.2)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_list)
    
    # 子图2: 测试集比例影响
    ax2 = axes[0, 1]
    test_sizes_label = [f"{r['test_size']*100:.0f}%" for r in test_size_results]
    accuracy_ts = [r['Accuracy'] for r in test_size_results]
    bars = ax2.bar(test_sizes_label, accuracy_ts, color='#A23B72', alpha=0.7)
    ax2.set_xlabel('测试集比例', fontsize=11)
    ax2.set_ylabel('准确率', fontsize=11)
    ax2.set_title('测试集比例对准确率的影响 (K=5)')
    ax2.set_ylim([min(accuracy_ts)-0.05, max(accuracy_ts)+0.05])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for bar, acc in zip(bars, accuracy_ts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 子图3: 热力图
    ax3 = axes[1, 0]
    im = ax3.imshow(accuracy_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(test_sizes)))
    ax3.set_yticks(range(len(k_values)))
    ax3.set_xticklabels([f'{ts*100:.0f}%' for ts in test_sizes])
    ax3.set_yticklabels([f'K={k}' for k in k_values])
    ax3.set_xlabel('测试集比例', fontsize=11)
    ax3.set_ylabel('K 值', fontsize=11)
    ax3.set_title('准确率热力图 (K值 vs 测试集比例)')
    
    # 在热力图上显示数值
    for i in range(len(k_values)):
        for j in range(len(test_sizes)):
            text = ax3.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('准确率', rotation=270, labelpad=15)
    
    # 子图4: 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    max_acc_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_k = k_values[max_acc_idx[0]]
    best_test_size = test_sizes[max_acc_idx[1]]
    best_accuracy = accuracy_matrix[max_acc_idx[0], max_acc_idx[1]]
    
    stats_text = f"""
    【最优超参数组合】
    K值: {best_k}
    测试集比例: {best_test_size*100:.0f}%
    最高准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
    
    【其他统计】
    准确率平均值: {accuracy_matrix.mean():.4f}
    准确率最小值: {accuracy_matrix.min():.4f}
    准确率最大值: {accuracy_matrix.max():.4f}
    准确率标准差: {accuracy_matrix.std():.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('knn_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("\n✓ 超参数调整图表已保存为 'knn_hyperparameter_tuning.png'")
    plt.show()


# ==================== 任务5.3: 分类问题评判指标详解 ====================

def task_5_3_classification_metrics(X_scaled, y):
    """
    任务5.3: 详细介绍分类问题的各种评判指标
    包括：准确率、精确率、召回率、F1分数、混淆矩阵、ROC-AUC等
    """
    print("\n" + "=" * 80)
    print("任务 5.3: 分类问题的评判指标详解")
    print("=" * 80)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]  # 获取正类概率
    
    # ===== 1. 混淆矩阵 (Confusion Matrix) =====
    print("\n【1. 混淆矩阵 (Confusion Matrix)】")
    print("─" * 80)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"""
混淆矩阵定义：
┌─────────────────┬──────────────┬──────────────┐
│                 │  预测为负(0) │  预测为正(1) │
├─────────────────┼──────────────┼──────────────┤
│ 实际为负(0)     │     TN       │     FP       │
│ 实际为正(1)     │     FN       │     TP       │
└─────────────────┴──────────────┴──────────────┘

当前模型的混淆矩阵：
    预测为 0    预测为 1
实际为 0   {cm[0,0]:3d}        {cm[0,1]:3d}
实际为 1   {cm[1,0]:3d}        {cm[1,1]:3d}

含义说明：
    TN (True Negative)   = {cm[0,0]:3d}  【真负】正确预测的负类样本
    FP (False Positive)  = {cm[0,1]:3d}  【假正】错误预测为正的负类样本
    FN (False Negative)  = {cm[1,0]:3d}  【假负】错误预测为负的正类样本
    TP (True Positive)   = {cm[1,1]:3d}  【真正】正确预测的正类样本
    """)
    
    # ===== 2. 准确率 (Accuracy) =====
    accuracy = accuracy_score(y_test, y_pred)
    print("\n【2. 准确率 (Accuracy - ACC)】")
    print("─" * 80)
    print(f"""
定义: 所有预测正确的样本占总样本的比例
公式: Accuracy = (TP + TN) / (TP + TN + FP + FN)

计算: ({cm[1,1]} + {cm[0,0]}) / ({cm[1,1]} + {cm[0,0]} + {cm[0,1]} + {cm[1,0]})
    = {cm[1,1] + cm[0,0]} / {cm[1,1] + cm[0,0] + cm[0,1] + cm[1,0]}
    = {accuracy:.4f} ({accuracy*100:.2f}%)

适用场景:
    ✓ 用于平衡数据集，各类别样本数量差不多
    ✗ 不适合不平衡数据集，易被多数类主导
    
特点:
    - 最直观的评估方式
    - 但容易受类别不平衡影响
    """)
    
    # ===== 3. 精确率 (Precision) =====
    precision = precision_score(y_test, y_pred)
    print("\n【3. 精确率 (Precision - P)】")
    print("─" * 80)
    print(f"""
定义: 预测为正的样本中，实际为正的样本比例
      (在所有"声称为患病"的人中，实际患病的比例)
公式: Precision = TP / (TP + FP)

计算: {cm[1,1]} / ({cm[1,1]} + {cm[0,1]})
    = {cm[1,1]} / {cm[1,1] + cm[0,1]}
    = {precision:.4f} ({precision*100:.2f}%)

含义: 
    假阳性(FP)降低 → 精确率升高
    即：假阳性越少，模型"宣称患病"的准确度越高

应用场景:
    ✓ 当假正例(FP)代价大时使用
      例如: 医疗诊断中的误诊患者(给健康人标签患病)
            垃圾邮件过滤中的假阳性(误把正常邮件当垃圾)
    
    ✓ 关注：在所有预测为正的样本中，有多少实际是正的
    """)
    
    # ===== 4. 召回率 (Recall) =====
    recall = recall_score(y_test, y_pred)
    print("\n【4. 召回率 (Recall - R) / 灵敏度 (Sensitivity)】")
    print("─" * 80)
    print(f"""
定义: 实际为正的样本中，被正确预测的比例
      (在所有"真正患病"的人中，被正确识别的比例)
公式: Recall = TP / (TP + FN)

计算: {cm[1,1]} / ({cm[1,1]} + {cm[1,0]})
    = {cm[1,1]} / {cm[1,1] + cm[1,0]}
    = {recall:.4f} ({recall*100:.2f}%)

含义:
    假阴性(FN)降低 → 召回率升高
    即：假阴性越少，对正类样本的"覆盖"越全面

应用场景:
    ✓ 当假负例(FN)代价大时使用
      例如: 医疗诊断中的漏诊患者(把患病的人当成健康人)
            癌症筛查中的漏诊(放过真正患癌患者)
    
    ✓ 关注：在所有实际为正的样本中，有多少被正确识别
    """)
    
    # ===== 5. F1分数 (F1-Score) =====
    f1 = f1_score(y_test, y_pred)
    print("\n【5. F1分数 (F1-Score)】")
    print("─" * 80)
    print(f"""
定义: Precision 和 Recall 的调和平均数
      平衡考虑假正例和假负例的影响
公式: F1 = 2 × (Precision × Recall) / (Precision + Recall)

计算: 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})
    = {f1:.4f} ({f1*100:.2f}%)

特点:
    • 当 Precision 和 Recall 差异大时，F1分数会降低
    • 更适合数据不平衡的场景
    • 值域：0~1，越接近1越好

应用场景:
    ✓ 用于平衡关注 FP 和 FN 的场景
    ✓ 处理不平衡数据集时的推荐指标
    ✓ 需要综合考虑精确率和召回率
    """)
    
    # ===== 分类报告 =====
    print("\n【分类详细报告】")
    print("─" * 80)
    print(classification_report(y_test, y_pred, target_names=['无糖尿病(0)', '有糖尿病(1)']))
    
    # ===== 6. ROC曲线和AUC值 =====
    if len(np.unique(y)) == 2:  # 二分类
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        print("\n【6. ROC曲线 和 AUC值】")
        print("─" * 80)
        print(f"""
ROC (Receiver Operating Characteristic) 曲线：
    • 横轴: 假正类率 (FPR) = FP / (FP + TN)
    • 纵轴: 真正类率 (TPR) = TP / (TP + FN) = Recall
    • 每个阈值对应曲线上一个点

AUC (Area Under Curve) 值：
    • ROC曲线下面积
    • 范围: 0~1
    • 含义: 
      - AUC=1.0: 完美分类器
      - AUC=0.5: 随机分类（对角线）
      - AUC<0.5: 比随机还差
    
当前模型的 AUC 值: {auc:.4f} ({auc*100:.2f}%)

特点:
    ✓ 不受样本类别不平衡影响
    ✓ 考虑了不同的分类阈值
    ✓ 综合评估模型性能
    """)
    
    # ===== 指标对比总结 =====
    print("\n【指标对比总结】")
    print("─" * 80)
    print(f"""
┌──────────────┬─────────────┬──────────────────────────────────────┐
│   指标名称   │   数值      │             适用场景                 │
├──────────────┼─────────────┼──────────────────────────────────────┤
│ 准确率(ACC)  │ {accuracy:.4f} │ 平衡数据集，整体分类效果评估      │
│ 精确率(P)    │ {precision:.4f} │ 控制假正例，如垃圾邮件过滤        │
│ 召回率(R)    │ {recall:.4f} │ 控制假负例，如医疗诊断              │
│ F1分数       │ {f1:.4f} │ 数据不平衡，综合评估                │
│ AUC值        │ {auc:.4f} │ 任何数据分布，全局性能评估        │
└──────────────┴─────────────┴──────────────────────────────────────┘
    """)
    
    # 可视化评估指标
    plot_classification_metrics(cm, accuracy, precision, recall, f1, 
                               fpr, tpr, auc, y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def plot_classification_metrics(cm, accuracy, precision, recall, f1, 
                                fpr, tpr, auc, y_test, y_pred_proba):
    """绘制分类评估指标的可视化图表"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('KNN分类问题 - 评判指标详解', fontsize=16, fontweight='bold')
    
    # 子图1: 混淆矩阵热力图
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
                xticklabels=['负例', '正例'], yticklabels=['负例', '正例'])
    ax1.set_title('混淆矩阵')
    ax1.set_ylabel('实际值')
    ax1.set_xlabel('预测值')
    
    # 子图2: 各指标柱状图
    ax2 = fig.add_subplot(gs[0, 1:])
    metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC值']
    values = [accuracy, precision, recall, f1, auc]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    bars = ax2.barh(metrics, values, color=colors)
    ax2.set_xlim([0, 1.0])
    ax2.set_xlabel('分数')
    ax2.set_title('各评估指标对比')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 在柱子上显示数值
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 子图3: ROC曲线
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC 曲线 (AUC={auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机分类器')
    ax3.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('假正类率 (FPR)')
    ax3.set_ylabel('真正类率 (TPR)')
    ax3.set_title('ROC曲线')
    ax3.legend(loc="lower right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 指标说明文本
    ax4 = fig.add_subplot(gs[1:, 2])
    ax4.axis('off')
    
    explanation = f"""
【混淆矩阵】
TN={cm[0,0]}, FP={cm[0,1]}
FN={cm[1,0]}, TP={cm[1,1]}

【各指标计算】
准确率 = (TP+TN)/(Total)
       = ({cm[1,1]}+{cm[0,0]})/{cm.sum()}
       = {accuracy:.4f}

精确率 = TP/(TP+FP)
       = {cm[1,1]}/({cm[1,1]}+{cm[0,1]})
       = {precision:.4f}

召回率 = TP/(TP+FN)
       = {cm[1,1]}/({cm[1,1]}+{cm[1,0]})
       = {recall:.4f}

F1 = 2×P×R/(P+R)
   = {f1:.4f}

AUC = {auc:.4f}
    """
    
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图5: 精确率和召回率权衡
    ax5 = fig.add_subplot(gs[2, 0])
    categories = ['精确率', '召回率']
    values_pr = [precision, recall]
    colors_pr = ['#FF6B6B', '#4ECDC4']
    bars = ax5.bar(categories, values_pr, color=colors_pr, alpha=0.7)
    ax5.set_ylim([0, 1.0])
    ax5.set_ylabel('分数')
    ax5.set_title('精确率 vs 召回率')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values_pr):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图6: 类别分布
    ax6 = fig.add_subplot(gs[2, 1])
    unique, counts = np.unique(y_test, return_counts=True)
    ax6.pie(counts, labels=['无糖尿病(0)', '有糖尿病(1)'], autopct='%1.1f%%',
           colors=['#90EE90', '#FFB6C6'], startangle=90)
    ax6.set_title('测试集类别分布')
    
    # 子图7: 预测概率分布
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='负例预测概率', color='#90EE90')
    ax7.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='正例预测概率', color='#FFB6C6')
    ax7.set_xlabel('预测为正的概率')
    ax7.set_ylabel('样本数')
    ax7.set_title('预测概率分布')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('classification_metrics.png', dpi=300, bbox_inches='tight')
    print("\n✓ 分类评估指标图表已保存为 'classification_metrics.png'")
    plt.show()


# ==================== 主函数 ====================

def main():
    """执行所有三个任务"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "Pima Indians 糖尿病数据集 - KNN分类分析" + " " * 19 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # 任务5.1: 基础KNN建模
    df, X_scaled, y = task_5_1_basic_knn()
    
    # 任务5.2: 超参数调整
    k_results, test_size_results = task_5_2_hyperparameter_tuning(X_scaled, y)
    
    # 任务5.3: 分类评判指标详解
    metrics = task_5_3_classification_metrics(X_scaled, y)
    
    # 最终总结
    print("\n" + "=" * 80)
    print("总结：KNN算法在Pima Indians糖尿病数据集上的应用")
    print("=" * 80)
    print("""
【关键学习点】：

1. 数据预处理
   • 特征标准化对KNN算法很重要（距离度量）
   • 处理缺失值和异常值

2. 超参数调优
   • K值的选择影响模型复杂度和泛化能力
   • 较小的K值：容易过拟合，模型复杂
   • 较大的K值：容易欠拟合，模型简单
   • 交叉验证可以更好地评估模型性能

3. 数据集划分
   • 训练/测试集比例影响评估效果
   • 典型比例：8:2 或 7:3
   • 确保两个集合都有足够样本量

4. 评判指标选择
   • 平衡数据：优先使用准确率
   • 关注FP代价大：使用精确率
   • 关注FN代价大：使用召回率
   • 不平衡数据：使用F1分数或AUC
   • 综合评估：使用ROC-AUC曲线

5. KNN算法特点
   ✓ 优点：简单易懂、无需训练、适合多分类
   ✗ 缺点：计算复杂度高、对特征尺度敏感、
          容易受异常值影响
    """)
    
    print("\n✓ 所有分析已完成！生成的图表已保存。")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
