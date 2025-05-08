import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import glob
import os
import time

print(mpl.get_cachedir())

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

names = ["10G"]

for name in names:
    start_time = time.time()
    dir = name + "_data_new"
    # 加载数据
    path = os.path.join("/data/qy/homework/2", dir, "part-*.parquet")

    files_10g = sorted(glob.glob(path))
    df_10g = pd.concat([pq.read_table(f, columns=['age', 'country', 'last_login']).to_pandas() for f in files_10g], ignore_index=True)
    # df_10g = pd.read_parquet(path)
    # 年龄分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df_10g['age'].dropna(), bins=30, kde=True)
    plt.title("Age Distribution_" + name)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig("age_dist_" + name + ".png")

    # 国家分布图
    plt.figure(figsize=(12, 6))
    df_10g['country'].value_counts().head(10).plot(kind='bar')
    plt.title("Top 10 Countries by User Count_" + name)
    plt.ylabel("User Count")
    plt.savefig("country_dist_" + name + ".png")


    # 确保 last_login 是 datetime 类型（会自动跳过格式错误的）
    df_10g['last_login'] = pd.to_datetime(df_10g['last_login'], errors='coerce')

    # 删除解析失败的（NaT）日期
    login_df = df_10g.dropna(subset=['last_login'])

    # 按月统计登录活跃用户数
    login_df['login_month'] = login_df['last_login'].dt.to_period('M')
    login_counts = login_df['login_month'].value_counts().sort_index()

    # 可视化
    plt.figure(figsize=(12, 6))
    login_counts.plot(kind='bar', color='skyblue')
    plt.title('每月登录用户数量')
    plt.xlabel('月份')
    plt.ylabel('登录次数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("login_" + name + ".png")

    end_time = time.time()  # 记录结束时间
    print(f"数据集 {name} 的程序运行时间：{end_time - start_time:.2f} 秒")
