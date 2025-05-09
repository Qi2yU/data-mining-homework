import pandas as pd
import json
import glob
import pyarrow.parquet as pq
from mlxtend.frequent_patterns import apriori, association_rules

# ---------- 读取 parquet 数据 ----------
def load_parquet_data(folder_path):
    files = sorted(glob.glob(folder_path + "/part-00000.parquet"))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    return df

# ---------- 提取商品类别交易 ----------
def extract_categories(row):
    try:
        items = json.loads(row)
        if isinstance(items, dict):  # 单一交易
            return [items.get('categories')]
        elif isinstance(items, list):  # 多商品订单
            return [item.get('categories') for item in items if 'categories' in item]
    except:
        return []

# ---------- 构建交易数据 ----------
def build_transaction_df(df):
    df['categories_list'] = df['purchase_history'].apply(extract_categories)
    transactions = df['categories_list'].dropna().tolist()
    transactions = [list(set(t)) for t in transactions if isinstance(t, list) and len(t) > 0]
    return transactions

# ---------- One-hot 编码 ----------
def transactions_to_df(transactions):
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

# ---------- 挖掘频繁项集 ----------
def run_apriori_analysis(oht_df, min_support=0.02, min_confidence=0.5):
    freq_itemsets = apriori(oht_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    return freq_itemsets, rules

# ---------- 分析含“电子产品”规则 ----------
def filter_rules_for_electronics(rules):
    return rules[rules['antecedents'].apply(lambda x: '电子产品' in x) |
                 rules['consequents'].apply(lambda x: '电子产品' in x)]

# ---------- 主程序 ----------
if __name__ == "__main__":
    # ⚠️ 替换为你的 parquet 数据路径（10G 或 30G）
    parquet_path = "/data/qy/homework/2/10G_data_new"  # 或 30G_data_new

    print("读取数据中...")
    df = load_parquet_data(parquet_path)

    print("提取交易（商品类别）...")
    transactions = build_transaction_df(df)

    print(f"共提取到 {len(transactions):,} 条有效交易。")

    print("One-hot 编码...")
    oht_df = transactions_to_df(transactions)

    print("执行 Apriori 算法...")
    freq_itemsets, rules = run_apriori_analysis(oht_df, min_support=0.02, min_confidence=0.5)

    print(f"共挖掘出 {len(rules)} 条关联规则。")

    print("筛选包含“电子产品”的规则...")
    electronics_rules = filter_rules_for_electronics(rules)

    print(electronics_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # 保存结果
    rules.to_csv("all_association_rules.csv", index=False)
    electronics_rules.to_csv("electronics_rules.csv", index=False)
