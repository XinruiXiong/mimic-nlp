#$ -N task1
#$ -V
#$ -cwd
#$ -o logs/task1/task1.o$JOB_ID
#$ -e logs/task1/task1.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 4
#$ -l gpu,cuda=1

papermill task1.ipynb task1_output.ipynb -k python3