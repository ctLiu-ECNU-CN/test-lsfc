import os
import pandas as pd

def merge_csv_files(folder_path):
    # 创建total文件夹（如果不存在）
    total_folder_path = os.path.join(folder_path, 'total')
    if not os.path.exists(total_folder_path):
        os.makedirs(total_folder_path)

    # 获取h和price文件夹中的所有CSV文件名
    close_folder_path = os.path.join(folder_path, 'close')
    price_folder_path = os.path.join(folder_path, 'price')
    low_folder_path = os.path.join(folder_path, 'low')
    high_folder_path = os.path.join(folder_path, 'high')
    h_files = os.listdir(close_folder_path)
    price_files = os.listdir(price_folder_path)
    low_files = os.listdir(low_folder_path)
    high_files = os.listdir(high_folder_path)

    # 遍历同名文件进行合并
    for filename in set(h_files).intersection(price_files, low_files, high_files):
        close_file_path = os.path.join(close_folder_path, filename)
        price_file_path = os.path.join(price_folder_path, filename)
        low_file_path = os.path.join(low_folder_path, filename)
        high_file_path = os.path.join(high_folder_path, filename)
        output_file_path = os.path.join(total_folder_path, filename)

        # 读取h、price、low和high的CSV文件
        close_data = pd.read_csv(close_file_path)
        price_data = pd.read_csv(price_file_path)
        low_data = pd.read_csv(low_file_path)
        high_data = pd.read_csv(high_file_path)

        # 合并四个DataFrame
        merged_data = pd.merge(price_data, close_data, on='Date')
        merged_data = pd.merge(merged_data, low_data, on='Date')
        merged_data = pd.merge(merged_data, high_data, on='Date')
        merged_data = merged_data.rename(columns={
            "Price":"Open",
            "Predictions_low_x": "Close",
            "Predictions_low_y": "Low",
            "Predictions_high": "High"
        })

        # 交换最大值和High, 最小值和Low
        for index, row in merged_data.iterrows():
            max_val = max(row['Open'], row['Close'], row['Low'], row['High'])
            min_val = min(row['Open'], row['Close'], row['Low'], row['High'])
            merged_data.at[index, 'High'] = max_val
            merged_data.at[index, 'Low'] = min_val
        # 保存合并后的数据到新的CSV文件
        merged_data.to_csv(output_file_path, index=False)

    print('合并完成！')

# 指定p文件夹的路径
folder_path = '../predictions_output'

# 调用函数进行合并
merge_csv_files(folder_path)