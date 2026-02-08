import pandas as pd

# 创建一个简单的DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}
df1 = pd.DataFrame(data)
print(df1)

# 创建两个Series对象
series_apples = pd.Series([1, 3, 7, 4], name="Apples")
series_bananas = pd.Series([2, 6, 3, 5], name="Bananas")

print(dir(series_apples))
print(series_bananas)

# 将两个Series对象相加，得到DataFrame，并指定列名
df2 = pd.DataFrame({ 'Apples': series_apples, 'Bananas': series_bananas })

# 显示DataFrame
print(df2)

df2.to_csv('fruit_sales.csv', index=False)  # 将DataFrame保存为CSV文件，不包含索引