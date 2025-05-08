import pandas as pd
import pyarrow.parquet as pq
import glob
import numpy as np
import json
import re
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

names = ["30G", "10G"]

for name in names:
    start_time = time.time()
    # ---------- 路径与数据加载 ----------
    dir = name + "_data_new"

    path = os.path.join("/data/qy/homework/2", dir, "part-00000.parquet")
    # path = os.path.join("/data/qy/homework/2", dir, "*.parquet")
    files = sorted(glob.glob(path))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    print('#' * 50, 'Dataset: ', name, '#' * 50)
    print(f"原始数据量：{len(df):,} 行")

    # ---------- 提取平均消费金额和消费次数 ----------
    def extract_purchase_features(purchase_str):
        try:
            data = json.loads(purchase_str)
            return pd.Series({
                'avg_purchase': data.get('avg_price', np.nan),
                'purchase_count': len(data.get('items', []))
            })
        except:
            return pd.Series({'avg_purchase': np.nan, 'purchase_count': 0})

    purchase_features = df['purchase_history'].apply(extract_purchase_features)
    df = pd.concat([df, purchase_features], axis=1)

    # ---------- 提取登录次数和最近登录时间 ----------
    def extract_login_features(login_str):
        try:
            data = json.loads(login_str)
            timestamps = pd.to_datetime(data.get('timestamps', []), errors='coerce')
            login_count = data.get('login_count', 0)
            last_login = max(timestamps) if len(timestamps) > 0 else pd.NaT
            return pd.Series({
                'login_count': login_count,
                'recent_login': last_login
            })
        except:
            return pd.Series({'login_count': 0, 'recent_login': pd.NaT})

    login_features = df['login_history'].apply(extract_login_features)
    df = pd.concat([df, login_features], axis=1)

    # ---------- 标准化评分（可换成更复杂的模型） ----------
    # 计算最近登录距今的天数
    today = pd.Timestamp(datetime.now().date())
    df['days_since_login'] = (today - df['recent_login']).dt.days.fillna(9999)

    # 将所有关键字段标准化（归一化处理）
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    df['score'] = (
        normalize(df['avg_purchase'].fillna(0)) * 0.4 +
        normalize(df['purchase_count']) * 0.2 +
        normalize(df['login_count']) * 0.2 +
        normalize(-df['days_since_login']) * 0.1 +
        df['is_active'] * 0.1
    )

    # ---------- 按照分数排序识别 Top 高价值用户 ----------
    df['value_segment'] = pd.qcut(df['score'], q=5, labels=['E', 'D', 'C', 'B', 'A'])  # 类似 RFM 分层
    high_value_users = df[df['value_segment'] == 'A']

    print(f"💎 高价值用户数量：{len(high_value_users)}")
    print(high_value_users[['id', 'user_name', 'avg_purchase', 'purchase_count', 'login_count', 'days_since_login', 'score']].head(10))

    # ---------- 可视化分数分布 ----------
    plt.figure(figsize=(10, 5))
    df['score'].hist(bins=50, color='orange')
    plt.title("用户价值评分分布")
    plt.xlabel("Score")
    plt.ylabel("用户数量")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("user_value_score_distribution_" + name + ".png")

    

