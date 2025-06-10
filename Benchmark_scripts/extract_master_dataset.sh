#$ -N mimic_master_dataset
#$ -V
#$ -cwd
#$ -o logs/mimic_master_dataset/mimic_master_dataset.o$JOB_ID
#$ -e logs/mimic_master_dataset/mimic_master_dataset.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 8

python extract_master_dataset.py --mimic4_path /u/scratch/x/xxiong/mimiciv/1.0 --output_path /u/scratch/x/xxiong/mimic-nlp-bench-new

# python extract_master_dataset.py --mimic4_path /u/scratch/x/xxiong/mimiciv/1.0 --output_path /u/scratch/x/xxiong/mimic-temp
