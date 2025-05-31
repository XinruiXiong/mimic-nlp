import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description='Generate discharge note embeddings using Bio_ClinicalBERT.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to input master_dataset.csv')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save output CSV with note embeddings')
parser.add_argument('--model_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT', help='HF model name')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding')
args = parser.parse_args()

# 模型与设备
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载数据
df = pd.read_csv(args.input_csv)
df['discharge_summary_text'] = df['discharge_summary_text'].fillna('')
text_list = df['discharge_summary_text'].tolist()

# 生成嵌入
all_embeddings = []
for i in tqdm(range(0, len(text_list), args.batch_size)):
    batch_texts = text_list[i:i+args.batch_size]
    encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state[:, 0, :].cpu()
    all_embeddings.append(embeddings)

# 合并并写入
all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
df_embed = pd.DataFrame(all_embeddings, columns=[f'note_embed_{i}' for i in range(all_embeddings.shape[1])])
df_embed = df_embed.set_index(df.index)
df_final = pd.concat([df, df_embed], axis=1)
df_final.to_csv(args.output_csv, index=False)
