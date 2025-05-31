#$ -N task2
#$ -V
#$ -cwd
#$ -o logs/task2/task2.o$JOB_ID
#$ -e logs/task2/task2.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 4
#$ -l gpu,cuda=1

papermill task2.ipynb task2_output.ipynb -k python3