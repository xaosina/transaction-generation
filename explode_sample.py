import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
import argparse

parser = argparse.ArgumentParser(description='Gensampels exploder')
parser.add_argument('-d', '--dataset', type=str, choices=["age", "alphabattle_small"], default='age')
parser.add_argument('-l', '--location', type=str, required=True) # example: 'evaluation_run/temp1.0_topk1'
parser.add_argument('--sort', dest='sort', action='store_const', const=True, default=False)

args = parser.parse_args()

DATA_TYPE = args.dataset
DATA_LOCATION = args.location
sort_by_time = args.sort


path = f'./log/generation/{DATA_TYPE}/{DATA_LOCATION}/seed_0/evaluation/samples/gen/part-0000.parquet' # change if needed
result_name = f'{DATA_TYPE}-gen-exploded.parquet' # change if needed

path = Path(path)
df_gen = pd.read_parquet(path)

def get_unroller(data_type: str = DATA_TYPE):
    if data_type == 'age':
        def unroller(row):
            d = dict()
            for k in row.keys():
                if k not in ['cliend_id', '_seq_len']:
                    d[k] = row[k]
            d['client_id'] = [row['client_id'],] * row['_seq_len']
            _df = pd.DataFrame(d)
            if sort_by_time:
                _df = _df.sort_values('trans_date', axis=0, kind='stable')
            return _df
    elif data_type == 'alphabattle_small':
        def unroller(row):
            d = dict()
            d['app_id'] = [row['app_id'],] * row['_seq_len']
            for k in row.keys():
                if k not in ['app_id', '_seq_len']:
                    d[k] = row[k]
            _df = pd.DataFrame(d)
            if sort_by_time:
                _df = _df.sort_values('hours_since_first_tx', axis=0, kind='stable')
            return _df
    
    return unroller

unroller = get_unroller(DATA_TYPE)

unrolled = (
    df_gen
    .apply(unroller, axis=1)
    .tolist()
)

df_unrolled = pd.concat(unrolled, ignore_index=True)
print('Head of the exploded df:\n', df_unrolled.head())
print('Tail of the exploded df:\n', df_unrolled.tail())
df_unrolled.to_parquet(result_name)
