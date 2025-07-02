import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 读取结果文件
with open("results_final.json", "r", encoding="utf-8") as f:
    result = json.load(f)

y_true = result["ground_truth"]
y_pred = [label if label is not None else "Center" for label in result["predict_label"]]

# 生成分类报告
report = classification_report(y_true, y_pred, output_dict=True)
report_text = classification_report(y_true, y_pred)

# 生成混淆矩阵
labels = sorted(list(set(y_true + y_pred)))  # 获取所有出现过的标签
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 输出文本报告
with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

report_df = pd.DataFrame(report).transpose()

print(report_df)

# 保存为 CSV 文件
report_df.to_csv("classification_report.csv", encoding="utf-8", index=True)
