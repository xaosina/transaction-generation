# Get ready for readable and easy to run code!!! 
#### 1. Generating transactions
#### 2. Evaluating generated transactions

### Setup
#### 1. Install requirements
> pip install -r requirements.txt

### Detection metric
#### 1. Idea
1. Take original data - add target 0
2. Take generated data - add target 1
3. Join and shuffle
4. Report GRU ROC-AUC on Cross Validation
#### 2. How to use
1. Preprocess data
> python preprocess/datafusion_detection.py --match-user -t tabsyn -n 16 -d log/generation/tabsyn/synth_dropped_2.csv -s data/detection/tabsyn_16
2. Run GRU. ATTENTION! Dont forget to add "/data" at the **end** of data path
> python main.py -t data/detection/tabsyn_16/data/ -e detection  --tqdm -d datafusion_detection

### Table metric
Table metrics measure how well generated data simulate each row individually.
1. Shape and trend score
> python evaluation/eval_density.py -d log/generation/tabsyn/synth_16.csv

See results in log/density/{name_of_file}.

### TCT metric
Evaluate how well mcc code intervals generated
> python evaluation/tct.py -d log/generation/tabsyn/synth_16.csv -n 16