#$ -N data_general_processing
#$ -V
#$ -cwd
#$ -o logs/data_general_processing/data_general_processing.o$JOB_ID
#$ -e logs/data_general_processing/data_general_processing.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 4



python data_general_processing.py \
  --master_dataset_path /u/scratch/x/xxiong/mimic-nlp-bench \
  --output_path /u/scratch/x/xxiong/mimic-nlp-bench