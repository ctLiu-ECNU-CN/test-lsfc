import pandas as pd
import matplotlib.pyplot as plt
stock_name = 'BYD'
# 读取 CSV 文件
df = pd.read_csv(f'../predictions_output/total/predictions{stock_name}.csv')

# 提取日期和所有列的数据
dates = pd.to_datetime(df['Date'])
columns = df.columns[1:]  # 排除日期列

# 创建图表
plt.figure(figsize=(10, 6))

# 遍历每一列，并绘制折线图
for column in columns:
    data = df[column]
    plt.plot(dates, data, label=column)

# 设置图例
plt.legend()

# 设置图表标题和坐标轴标签
plt.title('Data Variation')
plt.xlabel('Date')
plt.ylabel('Value')

# 旋转日期标签以避免重叠
plt.xticks(rotation=45)

# 显示图表
plt.show()