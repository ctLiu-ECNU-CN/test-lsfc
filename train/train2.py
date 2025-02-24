
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from models import Informer

# 数据集划分
train_size = 0.85
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 64
pre_len = 5
batch_size = 32
device = "cuda"
lr = 5e-5
epochs = 150

def create_data(datas):

    values = []
    labels = []

    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens-pre_len-s_len):
        value = datas[index:index+s_len, [0, 2, 3, 4, 5]]
        label = datas[index+s_len-pre_len:index+s_len+pre_len, [0, 1]]

        values.append(value)
        labels.append(label)

    return values, labels

def read_data():

    datas = pd.read_csv("../datas/BYD.csv")
    datas.pop("Adj Close")
    datas.fillna(0)

    xs = datas.values[:, [2, 3, 4, 5]]
    ys = datas.values[:, 1]

    x_stand.fit(xs)
    y_stand.fit(ys[:, None])

    values, labels = create_data(datas)

    train_x, test_x, train_y, test_y = train_test_split(values, labels, train_size=train_size)

    return train_x, test_x, train_y, test_y


# 自定义数据集
class RefactData(Dataset):
    def __init__(self, values, labels):

        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    def create_time(self, data):

        time = data[:, 0]
        time = pd.to_datetime(time)

        week = np.int32(time.dayofweek)[:, None]
        month = np.int32(time.month)[:, None]
        day = np.int32(time.day)[:, None]
        time_data = np.concatenate([month, week, day], axis=-1)

        return time_data

    def __getitem__(self, item):

        value = self.values[item]
        label = self.labels[item]

        value_t = self.create_time(value)
        label_t = self.create_time(label)

        value = x_stand.transform(value[:, 1:])
        label = y_stand.transform(label[:, 1][:, None])
        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t


def train():
    train_x, test_x, train_y, test_y = read_data()

    train_data = RefactData(train_x, train_y)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = RefactData(test_x, test_y)
    test_data = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    model = Informer(output_size=4)  # 修改output_size为4，表示预测开盘、收盘、最高和最低价格
    model.train()
    model.to(device)

    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(train_data)
        for step, (x, y, xt, yt) in enumerate(pbar):
            mask = torch.zeros_like(y)[:, pre_len:].to(device)

            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)

            logits = model(x, xt, dec_y, yt)

            loss = loss_fc(logits, y[:, pre_len:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            s = "train ==> epoch:{} - step:{} - loss:{}".format(epoch, step, loss)
            pbar.set_description(s)

        model.eval()
        with torch.no_grad():
            predictions = []
            for step, (x, y, xt, yt) in enumerate(test_data):
                mask = torch.zeros_like(y)[:, pre_len:].to(device)
                x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
                dec_y = torch.cat([y[:, :pre_len], mask], dim=1)

                logits = model(x, xt, dec_y, yt)

                # 从logits中提取预测结果
                final_prediction = logits[:, -pre_len:, :].cpu().numpy()

                # 反归一化预测结果
                final_prediction = y_stand.inverse_transform(final_prediction)

                # 将预测结果添加到列表中
                predictions.append(final_prediction)

            # 将所有预测结果连接成一个numpy数组
            predictions = np.concatenate(predictions)

        model.train()
        torch.save(model.state_dict(), "informer_model.pth")

if __name__ == '__main__':
    train()

