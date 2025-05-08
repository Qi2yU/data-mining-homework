import cudf
import cupy as cp
import glob
import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

names = ["30G", "10G"]

for name in names:
    start_time = time.time()

    # ---------- 路径与数据加载 ----------
    dir_path = os.path.join("/data/qy/homework/2", name + "_data_new")
    files = sorted(glob.glob(os.path.join(dir_path, "*.parquet")))

    df = cudf.concat([cudf.read_parquet(f) for f in files], ignore_index=True)
    print('#' * 50, 'Dataset: ', name, '#' * 50)
    print(f"原始数据量：{len(df):,} 行")

    # ---------- 提取 purchase 字段 ----------
    def extract_purchase(row):
        try:
            data = json.loads(row)
            avg_price = data.get("avg_price", cp.nan)
            item_count = len(data.get("items", []))
        except:
            avg_price = cp.nan
            item_count = 0
        return [avg_price, item_count]

    purchase_data = df['purchase_history'].to_pandas().apply(extract_purchase)
    purchase_df = cudf.DataFrame(purchase_data.tolist(), columns=['avg_purchase', 'purchase_count'])
    df = df.join(purchase_df)

    # ---------- 提取 login 字段 ----------
    def extract_login(row):
        try:
            data = json.loads(row)
            login_count = data.get("login_count", 0)
            timestamps = data.get("timestamps", [])
            last_login = max(timestamps) if timestamps else None
        except:
            login_count = 0
            last_login = None
        return [login_count, last_login]

    login_data = df['login_history'].to_pandas().apply(extract_login)
    login_df = cudf.DataFrame(login_data.tolist(), columns=['login_count', 'recent_login'])

    # recent_login 转换为 datetime
    login_df['recent_login'] = cudf.to_datetime(login_df['recent_login'], errors='coerce')
    df = df.join(login_df)

    # ---------- 评分打分 ----------
    today = cudf.to_datetime(str(datetime.today().date()))
    df['days_since_login'] = (today - df['recent_login']).dt.days.fillna(9999)

    def normalize(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-6)

    df['score'] = (
        normalize(df['avg_purchase'].fillna(0)) * 0.4 +
        normalize(df['purchase_count']) * 0.2 +
        normalize(df['login_count']) * 0.2 +
        normalize(-df['days_since_login']) * 0.1 +
        df['is_active'].astype('float32') * 0.1
    )

    # cuDF 不支持 qcut，转 Pandas 做 value_segment（或自定义）
    df_cpu = df.to_pandas()
    df_cpu['value_segment'] = pd.qcut(df_cpu['score'], q=5, labels=['E', 'D', 'C', 'B', 'A'])
    df = cudf.DataFrame.from_pandas(df_cpu)

    high_value_users = df[df['value_segment'] == 'A']
    print(f"💎 高价值用户数量：{len(high_value_users)}")
    print(high_value_users[['id', 'user_name', 'avg_purchase', 'purchase_count', 'login_count', 'days_since_login', 'score']].head(10))

    # ---------- 分数分布图（只能转到 CPU 再画） ----------
    plt.figure(figsize=(10, 5))
    df_cpu['score'].hist(bins=50, color='orange')
    plt.title("用户价值评分分布")
    plt.xlabel("Score")
    plt.ylabel("用户数量")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("user_value_score_distribution_" + name + ".png")

    print("运行时间：", round(time.time() - start_time, 2), "秒")
