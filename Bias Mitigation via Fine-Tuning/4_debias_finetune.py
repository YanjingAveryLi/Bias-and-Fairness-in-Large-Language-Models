
import os
import json
import random
import argparse
import hashlib
import pandas as pd
import openai
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, classification_report, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Bias Prediction Model Training and Evaluation")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")
    parser.add_argument("--data_dir", type=str, default = './data/json', help="Directory containing JSON data files")
    parser.add_argument("--output_dir", type=str, default = './data', help="Directory to save the output files")
    parser.add_argument("--test_file", type=str, default = "./data/data_public.csv",help="Test file path")
    parser.add_argument("--model_id", type=str, help="Fine-tuned model ID (optional)")
    return parser.parse_args()

def load_dataset(data_dir, label_filter=None):
    align_label = {'left': 'Left', 'center': 'Center', 'right': 'Right'}
    dataset = []
    files = os.listdir(data_dir)
    for file in files:
        if file == '.DS_Store':
            continue
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding='utf-8') as fcc_file:
            content_dict = json.load(fcc_file)
            content = content_dict['content'].strip()
            title = content_dict['title'].strip()
            ori_label = content_dict['bias_text']
            label = align_label[ori_label]
            all_content = title + content
            each_article = {"text": all_content.replace('\n', ''), "label": label}
            if label_filter is None or label == label_filter:
                dataset.append(each_article)
    return dataset

def save_dataset(dataset, output_file):
    data = {'data': dataset}
    with open(output_file, "w",encoding='utf-8') as f:
        json.dump(data, f)

def transform_jsonl(input_file_path, output_file_path):
    system_content = "You are an assistant that helps predict the political leaning of the input article. Reply with the media bias with Right, Center, or Left."
    entries = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)['data']
        for instance in data:
            entry = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": instance["text"]},
                    {"role": "assistant", "content": instance["label"]}
                ]
            }
            entries.append(entry)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in entries:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

def submit_and_fine_tune(client, file_path):
    upload_response = client.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    file_id = upload_response.id
    print(f"Uploaded file ID: {file_id}")
    job_response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": 3, "batch_size": 32, "learning_rate_multiplier": 2}
    )
    print(f"Fine-tuning job started. Job ID: {job_response.id}")
    return job_response

def normalize_bias(bias_text):
    bias_text = str(bias_text).lower()
    if "left" in bias_text:
        return "Left"
    elif "right" in bias_text:
        return "Right"
    elif "center" in bias_text:
        return "Center"
    else:
        return None

def evaluate_model(client, test_file_path, model_id, train_file_path=None):
    system_content = "You are an assistant that helps predict the political leaning of the input article. Reply with the media bias with Right, Center, or Left."

    # 步骤 1：读取训练集，生成文本哈希集合用于去重
    train_hashes = set()
    train_file_path = "./data/train_150.json"
    if train_file_path:
        with open(train_file_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)['data']
            train_hashes = {hashlib.md5(item["text"].strip().encode('utf-8')).hexdigest() for item in train_data}

    # 步骤 2：读取测试 CSV 数据
    df = pd.read_csv(test_file_path, encoding='utf-8', quotechar='"', on_bad_lines='skip')
    df["text"] = (df["title"].fillna('') + " " + df["body"].fillna('')).str.strip()
    df["label"] = df["bias"].apply(normalize_bias)
    df["hash"] = df["text"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())

    # 步骤 3：去除已在训练集中的数据
    df_filtered = df[~df["hash"].isin(train_hashes)]
    df_filtered = df_filtered[df_filtered["label"].notnull() & (df_filtered["text"].str.len() > 30)]

    # 步骤 4：从剩下的样本中随机抽取 300 条作为测试数据
    test_subset = df_filtered.sample(n=300, random_state=42) if len(df_filtered) >= 300 else df_filtered

    preds = []
    trues = []

    for _, row in test_subset.iterrows():
        text = row["text"]
        label = row["label"]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text[:15000]}
        ]

        try:
            response = client.chat.completions.create(model=model_id, messages=messages)
            pred = normalize_bias(response.choices[0].message.content.strip())
            trues.append(label)
            preds.append(pred)
            print(label, pred)
        except Exception as e:
            print(f"Error: {e}")
            with open('results_wrong.json', 'w', encoding='utf-8') as f:
                json.dump({"ground_truth": trues, "predict_label": preds}, f, ensure_ascii=False)

    # 保存最终预测结果
    with open('results_final.json', 'w', encoding='utf-8') as f:
        json.dump({"ground_truth": trues, "predict_label": preds}, f, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.api_key
    client = OpenAI()
    print('finished activating account')

    # 只构造一个训练集，每类各抽样 50 个样本
    dataset_left = random.sample(load_dataset(args.data_dir, 'Left'), 50)
    dataset_right = random.sample(load_dataset(args.data_dir, 'Right'), 50)
    dataset_center = random.sample(load_dataset(args.data_dir, 'Center'), 50)
    final_train_set = dataset_left + dataset_right + dataset_center
    print('finished building training dataset')


    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "train_150.json")
    jsonl_path = os.path.join(args.output_dir, "train_150.jsonl")
    save_dataset(final_train_set, json_path)
    transform_jsonl(json_path, jsonl_path)
    print('finished saving training dataset')


    # Fine-tune 一个模型
    print('started finetuning...')
    job_response = submit_and_fine_tune(client, jsonl_path)
    print('finished finetuning')


    # 可选：如果提供了模型 ID，立即评估
    if args.model_id:
        evaluate_model(client, args.test_file, args.model_id)
    else:
        print("Model fine-tuning started. Please wait for completion and then rerun with --model_id to evaluate.")
