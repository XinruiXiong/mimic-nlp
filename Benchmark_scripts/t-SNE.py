import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from MulticoreTSNE import MulticoreTSNE as TSNE  # 替换部分

# ---------- 1. Argument parsing ----------
parser = argparse.ArgumentParser(description='t-SNE visualization of note embeddings with MulticoreTSNE.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV with note embeddings')
parser.add_argument('--output_png', type=str, required=True, help='Path to save t-SNE plot image')
parser.add_argument('--n_pca', type=int, default=50, help='Number of PCA components before t-SNE')
parser.add_argument('--n_jobs', type=int, default=8, help='Number of CPU threads to use')
args = parser.parse_args()

# ---------- 2. Load embedding columns ----------
df = pd.read_csv(args.input_csv)
embed_cols = [col for col in df.columns if col.startswith('note_embed_')]
embeddings = df[embed_cols].values

# ---------- 3. PCA preprocessing ----------
scaler = StandardScaler()
embeddings_std = scaler.fit_transform(embeddings)

pca = PCA(n_components=args.n_pca, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_std)

# ---------- 4. t-SNE (Multicore) ----------
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, n_jobs=args.n_jobs, verbose=1, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# ---------- 5. Plot and save ----------
plt.figure(figsize=(10, 8))
plt.scatter(
    embeddings_tsne[:, 0],
    embeddings_tsne[:, 1],
    c=df['outcome_ed_revisit_3d'],  #coloring
    cmap='coolwarm',
    s=1,
    alpha=0.6
)
plt.title("t-SNE of Note Embeddings (PCA-50 → Multicore-tSNE-2)")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(args.output_png, dpi=300)
