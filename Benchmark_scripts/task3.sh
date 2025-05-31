#$ -N task3
#$ -V
#$ -cwd
#$ -o logs/task3/task3.o$JOB_ID
#$ -e logs/task3/task3.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 4
#$ -l gpu,cuda=1

papermill task3.ipynb task3_output.ipynb -k python3