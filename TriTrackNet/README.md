# 📘 TriTrackNet: A Dual-Channel Time Series Forecasting Model

TriTrackNet is a dual-channel time series forecasting model with multi-path interaction and perturbation-based optimization.  
It integrates **long-range temporal dependencies** (main channel) and **cross-dimensional interactions** (auxiliary channel), while using **PerturbOpt** to enhance robustness.  

This repository provides the implementation and benchmark experiments on several widely used datasets (ETT, Weather, Traffic, Electricity).  

---

## 🚀 Features
- Dual-channel architecture (main + auxiliary path)  
- Multi-path attention for temporal and feature dependencies  
- PerturbOpt adversarial optimizer for robustness  
- Extensive experiments on 7 benchmark datasets  

---

## 📂 Project Structure
```
TriTrackNet/
│
├── dataset/                # Benchmark datasets (.csv format)
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── WTH.csv
│   ├── ECL.csv
│   ├── traffic.csv
│   └── exchange.csv
│
├── TriTrackNet/            # Model code
│   ├── TriTrackNet.py      # Main model
│   └── utils/
│       ├── attention.py    # Attention modules
│       ├── dataset.py      # Data loader
│       ├── perturbopt.py   # PerturbOpt optimizer
│       ├── revin.py        # RevIN module
│       └── __init__.py
│
├── run_ECL.py              # Run script for Electricity dataset
├── run_Traffic.py          # Run script for Traffic dataset
├── run_WTH.py              # Run script for Weather dataset
├── run_ETTm1.py            # Run script for ETTm1 dataset
├── run_ETTm2.py            # Run script for ETTm2 dataset
├── run_ETTh1.py            # Run script for ETTh1 dataset
├── run_ETTh2.py            # Run script for ETTh2 dataset
├── run_exchange.py         # Run script for Exchange dataset
│
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/yourname/TriTrackNet.git
cd TriTrackNet

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Datasets
The following benchmark datasets are included in the `dataset/` folder:
- **ETT (ETTh1, ETTh2, ETTm1, ETTm2)** – Oil temperature related time series  
- **Weather** – Meteorological variables  
- **Electricity (ECL)** – Regional load series  
- **Traffic** – Road traffic data  
- **Exchange** – Exchange rates  

---

## ▶️ Running Experiments
Each dataset has its own run script. For example:

```bash
# Run on ETTh1 dataset
python run_ETTh1.py

# Run on Weather dataset
python run_WTH.py

# Run on Traffic dataset
python run_Traffic.py
```

All scripts will automatically load the corresponding dataset from `dataset/` and train/test TriTrackNet.  

---

## 📈 Results
Our experimental results demonstrate that **TriTrackNet** consistently outperforms baseline models such as Informer, Autoformer, iTransformer, PatchTST, and DLinear, especially on high-dimensional datasets (Traffic, Electricity).  
See the paper for full benchmark results.  


---

## 📬 Contact
For questions, please contact: **[your email or GitHub issues]**
