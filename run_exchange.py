import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from TriTrackNet.TriTrackNet import TriTrackNet


def read_ETTm1_dataset(seq_len, pred_len, time_increment=1):
    file_name ='E:\\TriTrackNet\\dataset\\exchange.csv'
    df_raw = pd.read_csv(file_name, index_col=0)
    n = len(df_raw)

    # 时间间隔为1分钟：一个小时有60分钟，一天有1440分钟（24 * 60）
    minutes_per_day = 24 * 60

    # 训练、验证、测试集划分：以分钟为单位
    train_days = 12 * 30  # 12个月
    val_days = 4 * 30  # 4个月
    test_days = 4 * 30  # 4个月

    # 转换为分钟
    train_end = train_days * minutes_per_day  # 训练集结束位置
    val_end = train_end + val_days * minutes_per_day  # 验证集结束位置
    test_end = val_end + test_days * minutes_per_day  # 测试集结束位置

    # 检查划分是否超出总数据集的长度，并调整
    if test_end > n:
        available_minutes = n - (seq_len + pred_len)  # 总可用时间长度，减去窗口序列和预测长度
        train_end = int(available_minutes * 0.7)  # 70% 分配给训练集
        val_end = int(available_minutes * 0.85)  # 15% 分配给验证集
        test_end = available_minutes  # 剩余 15% 分配给测试集

    # 数据集划分
    train_df = df_raw[:train_end]
    val_df = df_raw[train_end - seq_len:val_end]
    test_df = df_raw[val_end - seq_len:test_end]

    # 使用训练集进行标准化
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]

    # 使用滑动窗口生成数据
    x_train, y_train = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment)
    x_val, y_val = construct_sliding_window_data(val_df, seq_len, pred_len, time_increment)
    x_test, y_test = construct_sliding_window_data(test_df, seq_len, pred_len, time_increment)

    # 转换为 float32 类型
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # 展平目标矩阵
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_val, y_test = flatten(y_train), flatten(y_val), flatten(y_test)

    # 返回
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    range_ = np.arange(0, n_samples, time_increment)
    x, y = list(), list()
    for i in range_:
        x.append(data[i:(i + seq_len)].T)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T)
    return np.array(x), np.array(y)


if __name__ == '__main__':
    # read data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_ETTm1_dataset(seq_len=96, pred_len=96)
    # train model
    model = TriTrackNet(device='cuda', num_epochs=30, batch_size=256, base_optimizer=torch.optim.AdamW,
                      learning_rate=1e-3, weight_decay=1e-3, rho=0.9, use_revin=True)

    model.fit(x_train, y_train)
    # eval results
    y_pred_test = model.predict(x_test)

    # Calculate RMSE, MSE, and MAE
    rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    mse = np.mean((y_test - y_pred_test) ** 2)
    mae = np.mean(np.abs(y_test - y_pred_test))

    # Print the results
    print('RMSE:', rmse)
    print('MSE:', mse)
    print('MAE:', mae)
