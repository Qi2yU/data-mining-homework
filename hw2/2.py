import pandas as pd
import pyarrow.parquet as pq
import glob
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import matplotlib as mpl


print(mpl.get_cachedir())

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# ---------- 数据加载 ----------
path = "/data/qy/homework/2/10G_data_new/*.parquet"
files = sorted(glob.glob(path))[:2]
df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
print(f"读取数据中... 共 {len(df)} 条记录")

# ---------- 提取支付方式和商品类别 ----------
def extract_category_payment(purchase_json):
    try:
        data = json.loads(purchase_json)
        cat = data.get("categories", None)
        pay = data.get("payment_method", None)
        return (cat, pay, data.get("avg_price", 0))
    except:
        return (None, None, 0)

extracted = df['purchase_history'].apply(extract_category_payment)
df_extracted = pd.DataFrame(extracted.tolist(), columns=["category", "payment_method", "price"])
df_extracted.dropna(subset=["category", "payment_method"], inplace=True)

# ---------- 一、挖掘支付方式与商品类别之间的关联规则 ----------
# 为每条记录构建事务：将支付方式与类别视为“共现项”
transactions = df_extracted.apply(lambda row: [f"PAY_{row['payment_method']}", f"CAT_{row['category']}"], axis=1)

# 转换为布尔矩阵
te = TransactionEncoder()
te_ary = te.fit_transform(transactions.tolist())
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 挖掘频繁项集
freq_items = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# 仅保留“支付方式 → 类别”方向的规则
rules_filtered = rules[
    rules['antecedents'].apply(lambda x: any(i.startswith('PAY_') for i in x)) &
    rules['consequents'].apply(lambda x: any(i.startswith('CAT_') for i in x))
]

print(f"\n✅ 挖掘出 {len(rules_filtered)} 条支付方式与类别之间的有效关联规则（支持度≥0.01，置信度≥0.6）")
print(rules_filtered[["antecedents", "consequents", "support", "confidence", "lift"]].head())

# ---------- 二、高价值商品的首选支付方式 ----------
high_value = df_extracted[df_extracted["price"] > 5000]
method_counts = high_value["payment_method"].value_counts(normalize=True)

print("\n💰 高价值商品（价格 > 5000）首选支付方式占比:")
print(method_counts)

# 可视化高价值商品支付方式分布
method_counts.plot(kind='bar', color='skyblue')
plt.title("高价值商品的支付方式分布")
plt.xlabel("支付方式")
plt.ylabel("占比")
plt.tight_layout()
plt.savefig("high_value_payment_method.png")
plt.show()
