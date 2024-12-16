# Get ready for readable and easy to run code!!! 

# Setup

## **Using with package build**

Build the package

```bash
python -m pip install .
```

## Using without package build (not recommended)

1. Install the requirements

```bash
pip install -r requirements.txt
```

2. Copy `configs` directory to `evaluation` (i.e., there should exist directory `evaluation\configs`).

# Detection metric

#### Idea

1. Take original data - add target 0
2. Take generated data - add target 1
3. Join and shuffle
4. Report GRU ROC-AUC on Cross Validation

## **Using with package build**

From command line

```bash
python -m tmetrics.eval_detection --dataset datafusion -t general -d path/to/data/gen.csv -o path/to/data/orig.csv --gpu_ids 0 2 3 --verbose
```

From python file:

```python
from tmetrics.eval_detection import run_eval_detection
from pathlib import Path

data_type='general' # other option - `tabsyn`, it means
# additional preprocessing 
syn = Path('path/to/data/gen.csv')
orig = Path('path/to/data/orig.csv')
n_rows = 0 # length of generated/original sequences, 0 or -1
# for variable length (user_id's should be given in this case)
dataset='datafusion'

result = run_eval_detection(data_type, syn, orig, n_rows, match_users=False, dataset=dataset, gpu_ids=[0, 1], verbose=True)
```

## **Using without package build (not recommended)**


❗❗❗ (to be launched from repository's root directory)

```bash
python -m evaluation.eval_detection -t tabsyn -m -d log/generation/tabsyn/unet_16.csv -n 16 --verbose
```

# Datafusion-detection ML efficiency (age)

ML efficiency for age feature:
* train diffusion without age feature
* train a classifier (GRU) on generated data to predict GT age
* evaluate the trained classifier on GT data

## **Using with package build**

From command line

```bash
python -m tmetrics.eval_efficiency --dataset datafusion -t general -n 64 -d path/to/data/gen.csv -o path/to/data/orig.csv --gpu_ids 0 2 3 --verbose
```

From python file:

```python
from tmetrics.eval_efficiency import run_ml_efficiency
from pathlib import Path

tabsyn_preproc=False # if true - recovers `user_id` 
# and `days_since_first_tx`
syn = Path('path/to/data/gen.csv') # on this data the classifier is trained
orig = Path('path/to/data/orig.csv') # on this data the classifier is evaluated
n_rows = 64 # length of generated/original sequences 
# on which GRU classifier is trained
dataset='datafusion'

result = run_ml_efficiency(tabsyn_preproc, syn, orig, n_rows, sample_size=-1, dataset=dataset, gpu_ids=[0, 1], verbose=True)
```

## **Using without package build (not recommended)** 

TBD

# Shape and trend score
Table metrics measure how well generated data simulate each row individually.

## **Using with package build**

From command line

```bash
python -m tmetrics.eval_density -d tabsyn-concat/synthetic/datafusion_with_id/ae_train_32.csv -o tabsyn-concat/data/datafusion/preprocessed_with_id_test.csv
```

From python file:

```python
from tmetrics.eval_density import run_eval_density
from pathlib import Path

syn = Path('synthetic/datafusion_with_id/ae_train_16.csv')
orig = Path('data/datafusion/preprocessed_with_id_test.csv')

result = run_eval_density(syn, orig)
```
## **Using without package build (not recommended)** 

❗❗❗ (to be launched from repository's root directory)

```bash
python -m evaluation.eval_density -d log/generation/tabsyn/synth_16.csv
```

See results in log/density/{name_of_file}.

# TCT metric

## **Using without package build (not recommended)** 

❗❗❗ (to be launched from repository's root directory)

```bash
python -m evaluation.eval_tct -o data/datafusion_with_id/preprocessed_with_id_test.csv -d synthetic/datafusion_with_id/synth_64_099.csv --recover-len 64 --subsample-len 16,32,0 --seed 0,1,2,3,4
```

## **Using with package build**

From command line

```bash
python -m tmetrics.eval_tct --recover-len 32 -d tabsyn-concat/synthetic/datafusion_with_id/ae_train_32.csv -o tabsyn-concat/data/datafusion/preprocessed_with_id_test.csv --seed 0,1,2 --subsample-len 16,0
```

From python file:

```python
from tmetrics.eval_tct import run_eval_tct
from pathlib import Path

syn = Path('synthetic/datafusion_with_id/ae_train_32.csv')
orig = Path('data/datafusion/preprocessed_with_id_test.csv')
rec_len = 32
sbs_len = [16, 0] # 0 means 32
seeds = [0, 1, 2] # average ove three seeds

result = run_eval_tct(syn, orig, rec_len, sbs_len, seeds=seeds)
```
