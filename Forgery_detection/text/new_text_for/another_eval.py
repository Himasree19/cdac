import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =====================
# LOAD MODEL
# =====================
MODEL_PATH = r"C:\Users\LENOVO\Desktop\new_text_for\model"
DATA_DIR = r"C:\Users\LENOVO\Desktop\new_text_for\another\rf"
MAX_LEN = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)
model.eval()


def fix_column(df):
    col = df.columns[0]
    return df.rename(columns={col: 'text'})


def load_dataset(data_dir):
    if not os.path.isdir(data_dir):
        raise ValueError(f"Folder not found: {data_dir}")

    human_file = None
    ai_file = None

    files = os.listdir(data_dir)

    for file in files:
        fname = file.lower()
        if "human" in fname or "real" in fname:
            human_file = file
        elif "ai" in fname or "fake" in fname:
            ai_file = file

    if human_file is None or ai_file is None:
        raise ValueError(
            f"Could not find real/fake csv files in: {data_dir}. "
            f"Files found: {files}"
        )

    human_df = fix_column(pd.read_csv(os.path.join(data_dir, human_file)))
    ai_df = fix_column(pd.read_csv(os.path.join(data_dir, ai_file)))

    human_df["label"] = 0
    ai_df["label"] = 1

    return pd.concat([human_df, ai_df]).reset_index(drop=True)


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    return pred


df = load_dataset(DATA_DIR)

all_preds = []
all_labels = []

for i in tqdm(range(len(df))):
    text = df.iloc[i]['text']
    label = df.iloc[i]['label']

    pred = predict(text)

    all_preds.append(pred)
    all_labels.append(label)

acc = accuracy_score(all_labels, all_preds)
print(f"Another Dataset Accuracy: {acc*100:.2f}%")

labels_map = {0: "Real", 1: "Fake"}
labels_true = [labels_map[l] for l in all_labels]
preds_named = [labels_map[p] for p in all_preds]

cm = confusion_matrix(labels_true, preds_named)
classes = ['Real', 'Fake']

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Another Dataset Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.show()
