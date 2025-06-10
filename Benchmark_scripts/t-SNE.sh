#$ -N t-SNE
#$ -V
#$ -cwd
#$ -o logs/t-SNE/t-SNE.o$JOB_ID
#$ -e logs/t-SNE/t-SNE.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 4

python t-SNE.py \
  --input_csv /u/scratch/x/xxiong/mimic-nlp-bench-new/master_dataset_with_note_embed.csv \
  --output_png ./tsne_out/tsne_plot_outcome_ed_revisit_3d.png \
  --n_pca 50 \
  --n_jobs 4


# python t-SNE.py \
#   --input_csv /u/scratch/x/xxiong/mimic-nlp-bench/master_dataset_with_note_embed.csv \
#   --output_png ./tsne_out/tsne_plot_hospital_previous.png \
#   --n_pca 50 \
#   --n_jobs 4