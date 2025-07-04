# mimic-nlp
use mimic-iv, mimic-iv-ed, mimic-iv-note to perform 3 clinical prediction task


### This work is based on the work from [niulab](https://github.com/nliulab/mimic4ed-benchmark)


### 0. Data Preparation

MIMIC-IV-ED can be downloaded from [https://physionet.org/content/mimic-iv-ed/1.0/](https://physionet.org/content/mimic-iv-ed/1.0/) 

MIMIC-IV can be downloaded from [https://physionet.org/content/mimiciv/1.0/](https://physionet.org/content/mimiciv/1.0/)

MIMIC-IV-Note can be downloaded from [https://www.physionet.org/content/mimic-iv-note/2.2/](https://www.physionet.org/content/mimic-iv-note/2.2/)

***NOTE** It should be noted that upon downloading and extracting the MIMIC databases from their compressed files, the directory `/mimic-iv-ed-1.0/ed` and should `/mimic-iv-ed-1.0/note` be moved/copied to the directory containing MIMIC-IV data `/mimic-iv-1.0`.

### 1. Set-up
1. Clone this repository
```bash
git clone https://github.com/XinruiXiong/mimic-nlp.git
cd mimic-nlp
```
2. Install Package
```Shell
conda create -n mimic-bench python=3.11
conda activate mimic-bench
pip install -r requirements.txt
```

### 2. Benchmark Data Generation
~~~
python extract_master_dataset.py --mimic4_path {mimic4_path} --output_path {output_path}
~~~

### 3. NLP
~~~
python nlp_script.py \
  --input_csv {master_dataset_path} \
  --output_csv {output_path} \
  --model_name emilyalsentzer/Bio_ClinicalBERT \
  --window_size 512 \
  --stride 256 \
  --batch_size 16
~~~

You may change argument if you want.

### 3.5 Embeddings evaluation
~~~
python t-SNE.py \
  --input_csv {NLPed_master_dataset_path} \
  --output_png ./tsne_out/tsne_plot_outcome_ed_revisit_3d.png \
  --n_pca 50 \
  --n_jobs 4
~~~
You may change argument in the py file to color it with different prediction targets.

### 4. Cohort Filtering and Data Processing
~~~
python data_general_processing.py --master_dataset_path {NLPed_master_dataset_path} --output_path {output_path}
~~~

### 5. Prediction Task Selection and Model evaluation

~~~
papermill task{i}.ipynb task1_output.ipynb -k python3
~~~

change i to 1,2,3.




