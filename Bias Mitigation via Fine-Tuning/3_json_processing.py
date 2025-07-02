import pandas as pd
import os
import json

df = pd.read_csv("./data/data_public.csv",quotechar='"',
    encoding="utf-8",
    on_bad_lines='skip')

# 映射 bias 字段
def normalize_bias(bias):
    bias = str(bias).lower()
    if "left" in bias:
        return "left"
    elif "right" in bias:
        return "right"
    elif "center" in bias or "centre" in bias:
        return "center"
    else:
        return None

df['bias_text'] = df['bias'].apply(normalize_bias)
df = df.dropna(subset=['bias_text'])

# 遍历每一行，保存成单独的 JSON 文件
for idx, row in df.iterrows():
    item = {
        "title": str(row['title']) if pd.notnull(row['title']) else "",
        "content": str(row['body']) if pd.notnull(row['body']) else "",
        "bias_text": row['bias_text']
    }
    with open(f"./data/json/article_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(item, f, ensure_ascii=False, indent=2)

print("CSV 已成功转换为 JSON 格式，保存在 data_dir 文件夹中。")
