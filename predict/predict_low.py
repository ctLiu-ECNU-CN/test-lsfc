import os
import sys

sys.path.append('D:\workSpace\informer-Amazon-main')
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.model import Informer
import matplotlib.pyplot as plt

stock_name = "BYD"

# 加载训练好的模型
model = Informer()
model.load_state_dict(torch.load(f"../model_stock/low/informer_model_low_{stock_name}.pth"))  # 使用f-string
model.eval()

# 设置输出文件的位置
output_directory = "../predictions_output/low"
#os.makedirs(output_directory, exist_ok=True)

# 加载数据
data = pd.read_csv(f"../datas/{stock_name}.csv")
data.pop("Adj Close")  # 删除不必要的列
data.fillna(0, inplace=True)  # 填充缺失值（如果有的话）

# 定义数据缩放器
x_stand = StandardScaler()
y_stand = StandardScaler()

# 数据集划分
train_size = 0.85
s_len = 64
pre_len = 5


def create_time(data):
    time = pd.to_datetime(data.iloc[:, 0])  # 直接从 DataFrame 中提取日期时间并转换为 datetime 类型
    week = np.int32(time.dt.dayofweek)[:, None]  # 使用 dt 属性访问日期时间信息
    month = np.int32(time.dt.month)[:, None]
    day = np.int32(time.dt.day)[:, None]
    time_data = np.concatenate([month, week, day], axis=-1)
    return time_data


# 创建并拟合目标变量的缩放器
train_y = data.values[:, [1]]
y_stand.fit(train_y.reshape(-1, 1))

# 缩放输入数据
xs = data.values[:, [1, 2, 4, 5]]
xs = x_stand.fit_transform(xs)  # 缩放输入数据
xs = torch.tensor(xs, dtype=torch.float32).unsqueeze(0)  # 添加批次维度


# 进行预测
# 以下部分略有变化，需要根据实际情况进行调整

# 将数据保存为csv文件
def save_predictions_to_csv(dates, predictions, directory, filename):
    df = pd.DataFrame({'Date': dates, 'Predictions_low': predictions})
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)


#讲数据添加到已有的csv文件,基于Date合并
def add_predictions_to_csv(dates, predictions, output_directory, filename):
    filepath = os.path.join(output_directory, filename)
    # 构建新的DataFrame，包含日期和预测结果
    new_data = pd.DataFrame({'Date': dates, 'Low': predictions})

    # 读取现有的CSV文件
    existing_data = pd.read_csv(filepath)

    # 合并现有数据和新数据，基于日期进行合并
    merged_data = pd.concat([existing_data, new_data], axis=1, ignore_index= True)

    # 保存合并后的数据到CSV文件
    merged_data.to_csv(output_directory + filename, index=False)

with torch.no_grad():
    predictions = []
    true_values = []  # 存储真实值
    for i in range(len(data) - s_len - pre_len + 1):  # 扩展预测范围
        x = xs[:, i:i + s_len, :]
        xt = create_time(data.iloc[i:i + s_len])
        xt = torch.tensor(xt, dtype=torch.long).unsqueeze(0)  # 添加批次维度并将数据类型转换为Long
        mask = torch.zeros(1, pre_len, 1)  # 用于未来预测的掩码
        dec_y = torch.cat([torch.zeros(1, pre_len, 1), mask], dim=1)
        yt = create_time(data.iloc[i + s_len - pre_len:i + s_len + pre_len])
        yt = torch.tensor(yt, dtype=torch.long).unsqueeze(0)  # 添加批次维度并将数据类型转换为Long

        logits = model(x, xt, dec_y, yt)
        final_prediction = logits[:, -1].cpu().numpy()
        final_prediction = y_stand.inverse_transform(final_prediction)  # 反向转换预测结果
        predictions.append(final_prediction)

        if i + s_len + pre_len < len(data):
            true_value = data.iloc[i + s_len + pre_len][1]  # 获取真实值
            true_values.append(true_value)
        else:
            true_values.append(np.nan)  # 如果超出数据范围，用 NaN 填充
# 获取日期时间数据
dates = pd.to_datetime(data.iloc[s_len + pre_len - 1:, 0])  # 修改索引起始位置为 s_len+pre_len-1
# 如果需要，对预测结果进行后处理


# 将预测结果转换为 numpy 数组
predictions = np.concatenate(predictions).flatten()



# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(dates[:-15], true_values[:-15], label='True Values', color='red', linestyle='dashed')  # 除去最后15天的真实值曲线
plt.plot(dates, predictions, label='Predictions_low', color='green')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Predictions and True Values Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Save predictions to a CSV file in the specified directory
save_predictions_to_csv(dates, predictions, output_directory, f'predictions{stock_name}.csv')
# add_predictions_to_csv(dates, predictions, output_directory, f'predictions{stock_name}.csv')