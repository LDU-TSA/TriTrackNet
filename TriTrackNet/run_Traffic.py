import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from TriTrackNet.TriTrackNet import TriTrackNet


def read_ETTh1_dataset(seq_len, pred_len, time_increment=1):
    file_name = "E:\\TriTrackNet\\dataset\\Traffic.csv"
    df_raw = pd.read_csv(file_name, index_col=0)
    n = len(df_raw)
    # train-validation-test split for ETTh1
    train_end = 12 * 30 * 24
    val_end = train_end + 4 * 30 * 24
    test_end = val_end + 4 * 30 * 24
    train_df = df_raw[:train_end]
    val_df = df_raw[train_end - seq_len: val_end]
    test_df = df_raw[val_end - seq_len: test_end]
    # standardize by training set
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]

    # Convert the data to float32
    train_df = train_df.astype(np.float32)
    val_df = val_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    # apply sliding window
    x_train, y_train = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment)
    x_val, y_val = construct_sliding_window_data(val_df, seq_len, pred_len, time_increment)
    x_test, y_test = construct_sliding_window_data(test_df, seq_len, pred_len, time_increment)

    # flatten target matrices
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_val, y_test = flatten(y_train), flatten(y_val), flatten(y_test)

    # Convert the data to float32
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # return
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
    path_to_file = "dataset/ETTh1.csv"
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_ETTh1_dataset(seq_len=720, pred_len=336)

    # train model
    model = TriTrackNet(device='cuda', num_epochs=30, batch_size=4, base_optimizer=torch.optim.AdamW,
                      learning_rate=1e-3, weight_decay=1e-3, rho=0.9, use_revin=True)
    model.fit(x_train, y_train)

    # eval results
    y_pred_test = model.predict(x_test)
    print('RMSE:', np.sqrt(np.mean((y_test - y_pred_test) ** 2)))

    # 计算 MSE
    mse = np.mean((y_test - y_pred_test) ** 2)
    print('MSE:', mse)

    # 计算 MAE
    mae = np.mean(np.abs(y_test - y_pred_test))
    print('MAE:', mae)
