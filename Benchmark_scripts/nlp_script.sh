#$ -N clinical_notes_nlp
#$ -V
#$ -cwd
#$ -o logs/clinical_notes_nlp/clinical_notes_nlp.o$JOB_ID
#$ -e logs/clinical_notes_nlp/clinical_notes_nlp.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 8
#$ -l A100,gpu,gpu_mem=80G,cuda=0


# python nlp_script.py \
#   --input_csv /u/scratch/x/xxiong/mimic-nlp-bench/master_dataset.csv \
#   --output_csv /u/scratch/x/xxiong/mimic-nlp-bench/master_dataset_with_note_embed.csv \
#   --model_name emilyalsentzer/Bio_ClinicalBERT \
#   --batch_size 16


python nlp_script.py \
  --input_csv /u/scratch/x/xxiong/mimic-nlp-bench-new/master_dataset.csv \
  --output_csv /u/scratch/x/xxiong/mimic-nlp-bench-new/master_dataset_with_note_embed.csv \
  --model_name emilyalsentzer/Bio_ClinicalBERT \
  --window_size 512 \
  --stride 256 \
  --batch_size 16