# -*- coding: utf-8 -*-
"""
run_ECL_efficiency.py
一键评测 TriTrackNet(SAMFormer 封装) 在 Electricity (ECL) 数据集的计算开销。
- 6:2:2 时间序切分（若不足则退化 70%:15%:15%），滑窗生成
- 训练时间/推理时间/峰值显存/参数量/MACs
- 采用“子集测量 + 线性外推”避免内存爆炸（可选关闭外推）
- 自动探测 self.network.forward 的输入形状 (B,D,L) 或 (B,L,D)，保证 MACs 稳定计算
- 导出效率表（CSV + LaTeX）

依赖：
  pip install thop   # 或 pip install ptflops
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from TriTrackNet.TriTrackNet import TriTrackNet


# =========================
# 数据读取：ECL（电力），时间序切分 + 滑窗
# =========================
def read_ecl_dataset(path_csv: str, seq_len: int, pred_len: int, time_increment: int = 1):
    """
    读取 ECL.csv 并切分为 6:2:2；当数据长度不足时退化为 70%:15%:15%。
    验证/测试段首各回退 seq_len，避免滑窗跨界。
    返回：(x_train,y_train),(x_val,y_val),(x_test,y_test)；x/y 为 numpy。
    """
    df_raw = pd.read_csv(path_csv, index_col=0)
    n = len(df_raw)

    # 以“小时”为单位的 12:4:4 个月（≈ 6:2:2）
    train_end = 12 * 30 * 24
    val_end   = train_end + 4 * 30 * 24
    test_end  = val_end   + 4 * 30 * 24

    # 若长度不足，则退化为 70/15/15（保证滑窗可用范围）
    if test_end > n:
        available = n - (seq_len + pred_len)
        train_end = int(available * 0.70)
        val_end   = int(available * 0.85)
        test_end  = available

    train_df = df_raw[:train_end]
    val_df   = df_raw[train_end - seq_len: val_end]
    test_df  = df_raw[val_end   - seq_len: test_end]

    # 标准化（训练集拟合）
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]

    # 滑窗
    x_train, y_train = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment)
    x_val,   y_val   = construct_sliding_window_data(val_df,   seq_len, pred_len, time_increment)
    x_test,  y_test  = construct_sliding_window_data(test_df,  seq_len, pred_len, time_increment)

    # 展平 y：(N, D, H) → (N, D*H)
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_val, y_test = flatten(y_train), flatten(y_val), flatten(y_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def construct_sliding_window_data(data: np.ndarray, seq_len: int, pred_len: int, time_increment: int = 1):
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    idx = np.arange(0, max(n_samples, 0), time_increment)
    x, y = [], []
    for i in idx:
        x.append(data[i:(i + seq_len)].T)                        # (D, T)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T) # (D, H)
    return np.array(x), np.array(y)


# =========================
# 抓内部 nn.Module（你的封装里是 self.network）
# =========================
def get_core_module(wrapper: TriTrackNet) -> Optional[torch.nn.Module]:
    for name in ["network", "net", "model", "backbone", "core", "module"]:
        if hasattr(wrapper, name) and isinstance(getattr(wrapper, name), torch.nn.Module):
            return getattr(wrapper, name)
    return None


def count_params_m(core: Optional[torch.nn.Module]):
    if core is None:
        return "NA"
    return sum(p.numel() for p in core.parameters()) / 1e6


def estimate_macs_g(core: Optional[torch.nn.Module], sample_input: Optional[torch.Tensor]):
    """
    优先 thop，其次 ptflops；若失败或拿不到 core 则返回 "NA"。
    """
    if core is None or sample_input is None:
        return "NA"

    # 1) thop
    try:
        from thop import profile
        core.eval().to(sample_input.device)
        with torch.no_grad():
            macs, _ = profile(core, inputs=(sample_input,), verbose=False)
        return macs / 1e9
    except Exception:
        pass

    # 2) ptflops
    try:
        from ptflops import get_model_complexity_info
        core.eval().cpu()
        shape = tuple(sample_input.shape[1:])  # e.g. (D, L) or (L, D)
        macs_str, _ = get_model_complexity_info(core, shape, as_strings=True, print_per_layer_stat=False)
        val = macs_str.lower().replace('gmac', '').strip()
        return float(val)
    except Exception:
        return "NA"


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================
# 自动探测 forward 输入形状 & 构造 example_tensor（用于 MACs）
# =========================
def detect_forward_shape_and_build_example(core: Optional[torch.nn.Module],
                                           batch_size: int, D: int, L: int, device: str):
    """
    返回 (example_tensor, shape_tag)：'BDL' 或 'BLD' 或 'UNKNOWN'
    """
    if core is None:
        return None, "UNKNOWN"

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    core = core.to(dev).eval()

    # 尝试 (B, D, L)
    try:
        x_bdl = torch.randn(batch_size, D, L, device=dev).float()
        with torch.no_grad():
            _ = core(x_bdl)
        return x_bdl, "BDL"
    except Exception:
        pass

    # 尝试 (B, L, D)
    try:
        x_bld = torch.randn(batch_size, L, D, device=dev).float()
        with torch.no_grad():
            _ = core(x_bld)
        return x_bld, "BLD"
    except Exception:
        pass

    return None, "UNKNOWN"


# =========================
# 效率评测（避免 OOM 的子集方案 + 全量外推）
# =========================
def measure_epoch_time_and_mem(model: TriTrackNet, x_train: np.ndarray, y_train: np.ndarray,
                               repeats: int = 1, max_train_samples: Optional[int] = None,
                               estimate_full_epoch: bool = True):
    """
    用一个受限子集(默认 batch_size*4)测训练时间/峰值显存，避免 OOM。
    如果 estimate_full_epoch=True，则按“每批耗时”线性外推全量 epoch 时间。
    返回: (time_mean, time_std), mem_gb, est_full_epoch_time
    """
    bs = getattr(model, "batch_size", 256)
    if max_train_samples is None:
        max_train_samples = min(len(x_train), bs * 4)  # 可按需增减
    x_sub = x_train[:max_train_samples]
    y_sub = y_train[:max_train_samples]

    times, mems = [], []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        old_epochs = model.num_epochs
        model.num_epochs = 1  # 只跑 1 个 epoch 做效率表
        t0 = time.time()
        model.fit(x_sub, y_sub)
        _cuda_sync()
        t1 = time.time()
        model.num_epochs = old_epochs
        times.append(t1 - t0)
        if torch.cuda.is_available():
            mems.append(torch.cuda.max_memory_allocated() / (1024 ** 3))

    time_mean = float(np.mean(times))
    time_std  = float(np.std(times)) if len(times) > 1 else 0.0
    mem_gb    = float(np.max(mems)) if mems else 0.0

    # 估算全量 epoch 时间（线性按批次数外推）
    est_full_epoch_time = None
    if estimate_full_epoch:
        import math
        n_full   = len(x_train)
        n_sub    = len(x_sub)
        sub_batches  = max(1, math.ceil(n_sub / bs))
        full_batches = max(1, math.ceil(n_full / bs))
        per_batch    = time_mean / sub_batches
        est_full_epoch_time = per_batch * full_batches

    return (time_mean, time_std), mem_gb, est_full_epoch_time


def measure_infer_time_ms_per_1000(model: TriTrackNet, x_test: np.ndarray, repeats: int = 50):
    bs = getattr(model, "batch_size", 256)
    x_batch = x_test[:bs]  # numpy
    _ = model.predict(x_batch)  # warmup
    _cuda_sync()
    t0 = time.time()
    for _ in range(repeats):
        _ = model.predict(x_batch)
    _cuda_sync()
    t1 = time.time()
    ms_per_batch = (t1 - t0) * 1000.0 / repeats
    return ms_per_batch * (1000.0 / bs)


def benchmark_to_row(model_name: str, wrapper: TriTrackNet,
                     x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                     example_tensor_for_macs: Optional[torch.Tensor],
                     repeats_train: int = 1, repeats_infer: int = 50,
                     max_train_samples: Optional[int] = None,
                     use_estimated_full_epoch: bool = True):
    # 训练耗时/显存（子集测），并估算全量 epoch 时间
    (t_mean, _t_std), mem_gb, est_full = measure_epoch_time_and_mem(
        wrapper, x_train, y_train, repeats=repeats_train,
        max_train_samples=max_train_samples, estimate_full_epoch=True
    )
    # 推理耗时（每1000样本）
    infer_ms_per_1000 = measure_infer_time_ms_per_1000(wrapper, x_test, repeats=repeats_infer)
    # 参数/MACs
    core     = get_core_module(wrapper)
    params_m = count_params_m(core)
    macs_g   = estimate_macs_g(core, example_tensor_for_macs)

    epoch_time_s = est_full if (use_estimated_full_epoch and est_full is not None) else t_mean

    return {
        "Model": model_name,
        "Parameters": (f"{params_m:.2f} M" if isinstance(params_m, float) else "NA"),
        "MACs": (f"{macs_g:.2f} G" if isinstance(macs_g, float) else "NA"),
        "Max Mem.(MB)": f"{mem_gb * 1024:.1f}",
        "Epoch Time(s)": f"{epoch_time_s:.1f}",
        "Infer / 1000(ms)": f"{infer_ms_per_1000:.1f}",
    }


# =========================
# 主程序
# =========================
if __name__ == '__main__':
    # 路径与超参（把 ecl_csv 改成你的实际 ECL.csv 路径）
    ecl_csv  = r"E:\TriTrackNet\samformer-main33\samformer-main\dataset\ECL.csv"
    seq_len  = 512
    pred_len = 720

    # 读数据
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_ecl_dataset(ecl_csv, seq_len, pred_len)

    # 初始化训练器（先设 1 epoch 用于效率表；batch_size 可按显存调）
    model = TriTrackNet(device='cuda', num_epochs=1, batch_size=256,
                      base_optimizer=torch.optim.Adam, learning_rate=1e-3,
                      weight_decay=1e-3, rho=0.9, use_revin=True)

    # 先用一个小子集 fit 让 self.network 初始化（若已 fit 可注释掉）
    warm_n = min(len(x_train), model.batch_size)
    model.fit(x_train[:warm_n], y_train[:warm_n])

    # 自动探测 forward 输入形状 & 构造 MACs 示例输入
    core = get_core_module(model)
    B = getattr(model, "batch_size", 256)
    D = x_train.shape[1]  # 变量数
    L = x_train.shape[2]  # 序列长度
    example_tensor, shape_tag = detect_forward_shape_and_build_example(core, B, D, L, model.device)
    print(f"[Info] Detected forward shape: {shape_tag}")

    # 建议子集大小（避免 OOM）：batch_size * 4
    subset_n = B * 4

    # 跑效率表
    row = benchmark_to_row(
        "TriTrackNet (Ours)", model, x_train, y_train, x_test,
        example_tensor_for_macs=example_tensor,
        repeats_train=1, repeats_infer=50,
        max_train_samples=subset_n,            # ← 关键：限制训练子集规模，避免 OOM
        use_estimated_full_epoch=True          # ← 用线性外推估算全量 epoch 时间（可改 False）
    )
    df = pd.DataFrame([row], columns=[
        "Model", "Parameters", "MACs", "Max Mem.(MB)", "Epoch Time(s)", "Infer / 1000(ms)"
    ])

    print("\n==== Efficiency Table (ECL, seq_len=%d) ====\n" % seq_len)
    print(df.to_string(index=False))

    # 保存 CSV + LaTeX
    df.to_csv("efficiency_ecl_seq%d.csv" % seq_len, index=False)
    with open("efficiency_ecl_table.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))

    # —— 如需继续正式训练 + 误差评估，请取消注释 —— #
    # model.num_epochs = 10
    # model.fit(x_train, y_train)
    # y_pred_test = model.predict(x_test)
    # rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    # mse  = np.mean((y_test - y_pred_test) ** 2)
    # mae  = np.mean(np.abs(y_test - y_pred_test))
    # print(f"\nRMSE: {rmse:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}\n")
