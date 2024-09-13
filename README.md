# Get ready for readable and easy to run code!!! 
#### 1. Generating transactions
#### 2. Evaluating generated transactions



### Setup

**Using without package build**

1. Install the requirements

> pip install -r requirements.txt

2. Copy `configs` directory to `evaluation` (i.e., there should exist directory `evaluation\configs`).

**Using with package build**

Build the package

> python -m pip install .

### Detection metric

#### 1. Idea

1. Take original data - add target 0
2. Take generated data - add target 1
3. Join and shuffle
4. Report GRU ROC-AUC on Cross Validation

#### 2. How to use

**Using without package build**

(to be launched from repository's root directory)

> python -m evaluation.eval_detection -t tabsyn -m -d log/generation/tabsyn/unet_16.csv -n 16 --tqdm

**Using with package build**

From command line

> python -m tmetrics.eval_detection -t tabsyn -m -d log/generation/tabsyn/unet_16.csv -n 16 --tqdm

From python file:

```python
from tmetrics.eval_detection import prepare_and_detect
from pathlib import Path

data_type='tabsyn'
syn = Path('synthetic/datafusion_with_id/ae_train_16.csv')
orig = Path('data/datafusion/preprocessed_with_id_test.csv')
n_rows = 16 # length of generated/original sequences

prepare_and_detect(data_type, syn, orig, n_rows, match_users=True, tqdm=True)
```


### Table metric
Table metrics measure how well generated data simulate each row individually.

#### 1. Shape and trend score

**Using without package build** 

(to be launched from repository's root directory)

> python -m evaluation.eval_density -d log/generation/tabsyn/synth_16.csv

See results in log/density/{name_of_file}.

**Using with package build**

From command line

> python -m tmetrics.eval_density -d log/generation/tabsyn/synth_16.csv

From python file:

```python
from tmetrics.eval_density import run_eval_density
from pathlib import Path

syn = Path('synthetic/datafusion_with_id/ae_train_16.csv')
orig = Path('data/datafusion/preprocessed_with_id_test.csv')

run_eval_density(syn, orig)
```

### TCT metric


Evaluate how well mcc code intervals generated
> python evaluation/tct.py -d log/generation/tabsyn/synth_16.csv -n 16