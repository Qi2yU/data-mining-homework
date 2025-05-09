import pandas as pd
import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# ---------- 读取数据 ----------
df = pd.read_parquet("/data/qy/homework/2/30G_data_new/part-00000.parquet")
print("数据加载完成，共 {} 条记录".format(len(df)))

# ---------- 提取退款记录中的商品类别 ----------
def extract_refunded_categories(ph_json):
    try:
        items = json.loads(ph_json).get('items', [])
        status = json.loads(ph_json).get('payment_status', '').strip()
        if status in ['已退款', '部分退款']:
            return list(set([item.get('categories') for item in items if item.get('categories')]))
        else:
            return None
    except:
        return None

df['refund_categories'] = df['purchase_history'].apply(extract_refunded_categories)
refund_df = df[df['refund_categories'].notnull()].copy()
transactions = refund_df['refund_categories'].tolist()

print("💸 涉及退款的交易数：", len(transactions))

# ---------- 转换为 one-hot 编码 ----------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# ---------- Apriori 挖掘频繁项集 ----------
freq_items = apriori(df_trans, min_support=0.005, use_colnames=True)
print("✅ 找到频繁项集数：", len(freq_items))

# ---------- 计算关联规则 ----------
rules = association_rules(freq_items, metric='confidence', min_threshold=0.4)
rules = rules.sort_values(by='lift', ascending=False)

if len(rules) == 0:
    print("⚠️ 没有满足置信度和支持度条件的退款相关规则。")
else:
    print("📌 满足条件的规则如下（前 10 条）：")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # ---------- 可视化 ----------
    plt.figure(figsize=(8, 5))
    rules['support'].plot(kind='hist', bins=20, color='coral', edgecolor='black')
    plt.title("退款相关规则的支持度分布")
    plt.xlabel("Support")
    plt.ylabel("规则数量")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("refund_rule_support_distribution.png")
    plt.show()
