# import argparse
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
# import os

# # 解析命令行参数
# parser = argparse.ArgumentParser(description='Generate discharge note embeddings using Bio_ClinicalBERT.')
# parser.add_argument('--input_csv', type=str, required=True, help='Path to input master_dataset.csv')
# parser.add_argument('--output_csv', type=str, required=True, help='Path to save output CSV with note embeddings')
# parser.add_argument('--model_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT', help='HF model name')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding')
# args = parser.parse_args()

# # 模型与设备
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = AutoModel.from_pretrained(args.model_name)
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 加载数据
# df = pd.read_csv(args.input_csv)
# df['discharge_summary_text'] = df['discharge_summary_text'].fillna('')
# text_list = df['discharge_summary_text'].tolist()

# # 生成嵌入
# all_embeddings = []
# for i in tqdm(range(0, len(text_list), args.batch_size)):
#     batch_texts = text_list[i:i+args.batch_size]
#     encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     embeddings = model_output.last_hidden_state[:, 0, :].cpu()
#     all_embeddings.append(embeddings)

# # 合并并写入
# all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
# df_embed = pd.DataFrame(all_embeddings, columns=[f'note_embed_{i}' for i in range(all_embeddings.shape[1])])
# df_embed = df_embed.set_index(df.index)
# df_final = pd.concat([df, df_embed], axis=1)
# df_final.to_csv(args.output_csv, index=False)


import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ---------- 1. 参数解析 ----------
parser = argparse.ArgumentParser(description='Generate discharge note embeddings with sliding window.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to input master_dataset.csv')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save output CSV with note embeddings')
parser.add_argument('--model_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT', help='HuggingFace model name')
parser.add_argument('--window_size', type=int, default=512, help='Token window size')
parser.add_argument('--stride', type=int, default=256, help='Stride for sliding window')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for encoding')
args = parser.parse_args()

# ---------- 2. 设置环境 ----------
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 3. 加载模型 ----------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name)
model.eval().to(device)
if device.type == 'cuda':
    model.half()  # 启用 FP16 推理（如支持）

# ---------- 4. 加载数据 ----------
df = pd.read_csv(args.input_csv)
df['discharge_summary_text'] = df['discharge_summary_text'].fillna('')
texts = df['discharge_summary_text'].tolist()

# ---------- 5. 滑窗切分函数 ----------
def window_token_chunks(text, tokenizer, window_size, stride):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for start in range(0, len(tokens), stride):
        end = start + window_size
        if start >= len(tokens):
            break
        chunk = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
    return chunks

# ---------- 6. 自定义 Dataset ----------
# class SlidingNoteDataset(Dataset):
#     def __init__(self, texts, tokenizer, window_size, stride):
#         self.tokenizer = tokenizer
#         self.window_size = window_size
#         self.stride = stride
#         self.windows_per_note = [window_token_chunks(t, tokenizer, window_size, stride) for t in texts]

#     def __len__(self):
#         return len(self.windows_per_note)

#     def __getitem__(self, idx):
#         return self.windows_per_note[idx]

class SlidingNoteDataset(Dataset):
    def __init__(self, texts, tokenizer, window_size, stride):
        self.texts = texts
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(text)
        windows = []
        for start in range(0, len(tokens), self.stride):
            end = start + self.window_size
            if start >= len(tokens):
                break
            chunk = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
            windows.append(chunk_text)
        return windows


# ---------- 7. 嵌入生成函数 ----------
def encode_note_windows(batch_windows):
    batch_embeddings = []
    for windows in batch_windows:
        if not windows:
            batch_embeddings.append(torch.zeros(model.config.hidden_size))
            continue

        encoded = tokenizer(windows, padding=True, truncation=True, max_length=args.window_size,
                            return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(**encoded)
        cls_embeddings = output.last_hidden_state[:, 0, :]  # (num_windows, hidden_size)
        mean_embedding = cls_embeddings.mean(dim=0).detach().cpu()
        batch_embeddings.append(mean_embedding)
    return torch.stack(batch_embeddings)

# ---------- 8. 创建 DataLoader 并处理 ----------
def collate_windows(batch):
    return batch  # 保留为 List[List[str]]，避免默认拼接时报错或卡死

dataset = SlidingNoteDataset(texts, tokenizer, args.window_size, args.stride)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows)

all_embeddings = []
for batch_windows in tqdm(loader, desc='Encoding notes'):
    emb_batch = encode_note_windows(batch_windows)
    all_embeddings.append(emb_batch)

# ---------- 9. 拼接写出 ----------
all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
df_embed = pd.DataFrame(all_embeddings, columns=[f'note_embed_{i}' for i in range(all_embeddings.shape[1])])
df_embed = df_embed.set_index(df.index)

# Drop 原始 note 列后合并
df_final = pd.concat([df.drop(columns=['discharge_summary_text']), df_embed], axis=1)
df_final.to_csv(args.output_csv, index=False)
