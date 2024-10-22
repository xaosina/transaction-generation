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
python -m tmetrics.eval_detection --dataset mbd -t tabsyn -m -n 16 -d tabsyn-concat/synthetic/datafusion_with_id/ae_train_16.csv -o tabsyn-concat/data/datafusion/preprocessed_with_id_test.csv --gpu_ids 0 2 3 --verbose --time-process
```

From python file:

```python
from tmetrics.eval_detection import run_eval_detection
from pathlib import Path

data_type='tabsyn'
syn = Path('synthetic/datafusion_with_id/ae_train_16.csv')
orig = Path('data/datafusion/preprocessed_with_id_test.csv')
n_rows = 16 # length of generated/original sequences
time_process=True # whether to take time features when metric compute

result = run_eval_detection(data_type, syn, orig, n_rows, match_users=True, verbose=True, time_process=time_process)
```

## **Using without package build (not recommended)**


❗❗❗ (to be launched from repository's root directory)

```bash
python -m evaluation.eval_detection -t tabsyn -m -d log/generation/tabsyn/unet_16.csv -n 16 --verbose
```

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
