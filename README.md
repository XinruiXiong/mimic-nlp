# mimic-nlp
use mimic-iv, mimic-iv-ed, mimic-iv-note to perform 3 clinical prediction task


### 0. Data Preparation

MIMIC-IV-ED can be downloaded from [https://physionet.org/content/mimic-iv-ed/1.0/](https://physionet.org/content/mimic-iv-ed/1.0/) 

MIMIC-IV can be downloaded from [https://physionet.org/content/mimiciv/1.0/](https://physionet.org/content/mimiciv/1.0/)

MIMIC-IV-Note can be downloaded from [[https://physionet.org/content/mimiciv/1.0/](https://www.physionet.org/content/mimic-iv-note/2.2/)]

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
  --batch_size 16
~~~

You may change argument if you want.


### 4. Cohort Filtering and Data Processing
~~~
python data_general_processing.py --master_dataset_path {master_dataset_path} --output_path {output_path}
~~~

### 5. Prediction Task Selection and Model evaluation

~~~
papermill task{i}.ipynb task1_output.ipynb -k python3
~~~

change i to 1,2,3.

