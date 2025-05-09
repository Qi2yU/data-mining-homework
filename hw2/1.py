import pandas as pd
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pyarrow.parquet as pq
import glob
import os

# ---------- 数据加载 ----------
def load_and_flatten_purchase_history(parquet_path):
    print("正在加载数据...")
    df = pd.concat([pq.read_table(file).to_pandas() for file in glob.glob(parquet_path)], ignore_index=True)

    transactions = []
    for record in df["purchase_history"]:
        try:
            data = json.loads(record)
            if isinstance(data, dict):
                category = data.get("categories")
                if isinstance(category, list):
                    transactions.append(category)
                else:
                    transactions.append([category])
        except Exception:
            continue

    return transactions

# ---------- 频繁项集挖掘（Apriori） ----------
def mine_association_rules(transactions, min_support=0.02, min_confidence=0.5):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print("正在执行 Apriori 算法...")
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(f"共发现 {len(frequent_itemsets)} 个频繁项集")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by="lift", ascending=False)
    print(f"共生成 {len(rules)} 条关联规则")

    return frequent_itemsets, rules

# ---------- 主函数 ----------
if __name__ == '__main__':
    parquet_path = "/data/qy/homework/2/10G_data_new/*.parquet"  # 替换为你的路径

    transactions = load_and_flatten_purchase_history(parquet_path)
    frequent_itemsets, rules = mine_association_rules(transactions)

    # 保存结果
    frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
    rules.to_csv("association_rules.csv", index=False)

    # 输出部分结果
    print("前5个频繁项集：")
    print(frequent_itemsets.head())

    print("\n前5条关联规则：")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
