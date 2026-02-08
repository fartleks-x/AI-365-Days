# Bike Day 线性回归预测

简单说明：使用线性回归预测 `bike-day.csv` 中的每日租赁总量（`cnt`）。

运行步骤：

1. 创建虚拟环境并激活（可选）：

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行脚本：

```bash
python linear_regression_bike.py
```

输出：在 `output/` 目录下会生成 `linear_model.joblib`、`scaler.joblib`、`pred_vs_actual.png`。
