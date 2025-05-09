import pandas as pd
import pyarrow.parquet as pq
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
# ---------- 数据加载 ----------
path = "/data/qy/homework/2/10G_data_new/*.parquet"
files = sorted(glob.glob(path))
df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
print(f"加载 {len(df)} 条数据")

# ---------- 提取类别和购买时间 ----------
def extract_date_category(purchase_json):
    try:
        data = json.loads(purchase_json)
        return pd.Series({
            'category': data.get("categories", None),
            'purchase_date': data.get("purchase_date", None),
            'user_id': data.get("user_id", None)  # 如果有用户ID的话
        })
    except:
        return pd.Series({'category': None, 'purchase_date': None, 'user_id': None})

extracted = df['purchase_history'].apply(extract_date_category)
extracted.dropna(subset=['category', 'purchase_date'], inplace=True)
extracted['purchase_date'] = pd.to_datetime(extracted['purchase_date'], errors='coerce')
extracted.dropna(subset=['purchase_date'], inplace=True)

# ---------- 衍生时间字段 ----------
extracted['year'] = extracted['purchase_date'].dt.year
extracted['month'] = extracted['purchase_date'].dt.month
extracted['quarter'] = extracted['purchase_date'].dt.quarter
extracted['weekday'] = extracted['purchase_date'].dt.dayofweek  # 0=Monday

# ---------- 一、季节性模式（季度/月/星期） ----------
# 按月份统计所有类别的购买数量
monthly_count = extracted.groupby(['month', 'category']).size().unstack().fillna(0)
monthly_count.plot(kind='line', figsize=(12, 6), title='各商品类别的月度购买量')
plt.xlabel("月份")
plt.ylabel("购买次数")
plt.tight_layout()
plt.savefig("monthly_category_trend.png")
plt.show()

# 按季度统计
quarterly_count = extracted.groupby(['quarter', 'category']).size().unstack().fillna(0)
quarterly_count.plot(kind='bar', stacked=True, figsize=(12, 6), title='季度商品类别分布')
plt.xlabel("季度")
plt.ylabel("购买次数")
plt.tight_layout()
plt.savefig("quarterly_category_trend.png")
plt.show()

# 按星期统计
weekday_count = extracted.groupby(['weekday', 'category']).size().unstack().fillna(0)
weekday_count.plot(kind='line', figsize=(12, 6), title='每周商品购买分布')
plt.xlabel("星期（0=周一）")
plt.ylabel("购买次数")
plt.tight_layout()
plt.savefig("weekday_category_trend.png")
plt.show()

# ---------- 二、探索先后购买模式 ----------
# 为了做序列模式，需按用户 + 时间排序
if 'id' in df.columns:
    extracted['user_id'] = df['id']

extracted = extracted.dropna(subset=['user_id'])
extracted.sort_values(by=['user_id', 'purchase_date'], inplace=True)

# 构建“前后购买对”
sequential_pairs = []
grouped = extracted.groupby("user_id")["category"].apply(list)

for cat_list in grouped:
    for i in range(len(cat_list) - 1):
        pair = (cat_list[i], cat_list[i + 1])
        sequential_pairs.append(pair)

# 统计购买顺序的频率
pair_df = pd.DataFrame(sequential_pairs, columns=['from_category', 'to_category'])
top_pairs = pair_df.value_counts().reset_index(name='count')
top_pairs = top_pairs[top_pairs['count'] >= 20]  # 筛选出现较多的

# 可视化：热力图
pivot = top_pairs.pivot(index='from_category', columns='to_category', values='count').fillna(0)
if pivot.empty:
    print("⚠️ 没有足够的先后购买对满足频率条件，跳过热力图绘制。")
else:
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title("🧭 类别间购买顺序频率（前→后）")
    plt.xlabel("后买")
    plt.ylabel("先买")
    plt.tight_layout()
    plt.savefig("sequential_category_heatmap.png")
    plt.show()
