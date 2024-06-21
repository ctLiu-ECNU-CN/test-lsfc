import pandas as pd

# 读取csv文件1
df1 = pd.read_csv("./datas_source/2018stock.csv")

# 筛选特定股票名称的数据
specific_stock = "Amazon"
df2 = df1[df1['股票名称'] == specific_stock]

# 重新排列数据并提取需要的列
df2 = df2[['日期', '开盘', '最高', '最低', '收盘', '收盘', '成交量']]

# 重命名列名
df2.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

#按照日期升序排列
df2 = df2.sort_values(by='Date')

# 保存为csv文件2
df2.to_csv("./datas/aier.csv", index=False)