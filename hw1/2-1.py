import pandas as pd
import pyarrow.parquet as pq
import glob
import numpy as np
import json
import re
import os
import time
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
    print('#' * 25, 'Dataset: ', name, '#' * 25)
    print(f"原始数据量：{len(df):,} 行")

    # ---------- 缺失值统计 ----------
    missing_report = df.isnull().sum().to_frame(name='缺失值数量')
    missing_report['缺失率'] = (missing_report['缺失值数量'] / len(df)).round(4)
    print("\n缺失值统计：")
    print(missing_report[missing_report['缺失值数量'] > 0])

    # ---------- 异常值检测 ----------
    abnormal_age = (df['age'] < 0) | (df['age'] > 100)
    abnormal_income = (df['income'] < 0) | (df['income'] > 1e7)
    print("\n异常值统计：")
    print(f"age 异常：{abnormal_age.sum()} ({abnormal_age.mean():.2%})")
    print(f"income 异常：{abnormal_income.sum()} ({abnormal_income.mean():.2%})")

    # ---------- 缺失值处理 ----------
    df['age'].fillna(df['age'].median(), inplace=True)
    df['income'].fillna(df['income'].median(), inplace=True)
    df['gender'].fillna(df['gender'].mode().iloc[0], inplace=True)

    # ---------- 异常值处理 ----------
    age_median = df.loc[~abnormal_age, 'age'].median()
    df.loc[abnormal_age, 'age'] = age_median
    original_len = len(df)
    df = df[~abnormal_income]
    print(f"\n删除 income 异常记录数：{original_len - len(df):,}")

    # ---------- Email & Phone 格式校验 ----------
    def is_valid_email(email):
        if pd.isnull(email):
            return False
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", str(email)))

    def is_valid_phone(phone):
        if pd.isnull(phone):
            return False
        digits = re.sub(r'\D', '', str(phone))
        return 10 <= len(digits) <= 15

    df['email_valid'] = df['email'].apply(is_valid_email)
    df['phone_valid'] = df['phone_number'].apply(is_valid_phone)

    invalid_email_count = (~df['email_valid']).sum()
    invalid_phone_count = (~df['phone_valid']).sum()
    print(f"\n无效邮箱：{invalid_email_count} ({invalid_email_count / len(df):.2%})")
    print(f"无效手机号：{invalid_phone_count} ({invalid_phone_count / len(df):.2%})")

    # ---------- purchase_history 结构解析与检查 ----------
    def is_purchase_empty(purchase_str):
        try:
            obj = json.loads(purchase_str)
            items = obj.get('items', [])
            return not items  # 空列表
        except:
            return True  # 解析失败也当空处理

    df['empty_purchase'] = df['purchase_history'].apply(is_purchase_empty)
    purchase_empty_count = df['empty_purchase'].sum()

    # ---------- login_history 结构解析与检查 ----------
    def is_login_empty(login_str):
        try:
            obj = json.loads(login_str)
            timestamps = obj.get('timestamps', [])
            return not timestamps
        except:
            return True

    df['empty_login'] = df['login_history'].apply(is_login_empty)
    login_empty_count = df['empty_login'].sum()

    print(f"purchase_history 无记录用户：{purchase_empty_count} ({purchase_empty_count / len(df):.2%})")
    print(f"login_history 无记录用户：{login_empty_count} ({login_empty_count / len(df):.2%})")

    # 可选删除无行为用户
    df = df[~(df['empty_login'] & df['empty_purchase'])]

    # ---------- 类别字段标准化与编码 ----------
    df['gender'] = df['gender'].str.lower().map({'male': 'male', 'female': 'female', 'other': 'other'})
    df['gender'].fillna('unknown', inplace=True)
    gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
    df['is_active'] = df['is_active'].astype(int)
    df = pd.concat([df, gender_dummies], axis=1)

    # ---------- 最终摘要 ----------
    print(f"清洗后数据量：{len(df):,} 行")
    print("数据预处理完成。")

    # ---------- 保存结果（可选） ----------
    # df.to_parquet("10G_data_cleaned.parquet")
    end_time = time.time()  # 记录结束时间
    print(f"数据集 {name} 的程序运行时间：{end_time - start_time:.2f} 秒")
