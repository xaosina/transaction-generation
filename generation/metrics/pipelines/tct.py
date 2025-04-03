import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path


CONF_LEVEL = 0.01

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig",
        help="Path to orig dataset containing CSV files",
        default="data/datafusion/preprocessed_with_id_test.csv",
        type=Path,
    )
    parser.add_argument(
        "-d", "--data", help="generated data csv path", type=Path, required=True
    )
    parser.add_argument(
        "-n",
        "--n-rows",
        type=str,
        required=True
    )
    return parser.parse_args()

def test_for_exp(rvs):
    # https://stats.stackexchange.com/questions/32061/what-are-the-standard-statistical-tests-to-see-if-data-follows-exponential-or-no
    rvs = rvs[rvs > 0]
    n = len(rvs)
    if n == 1:
        return np.nan
    bn = 2 * n / (1 + (n + 1) / (6 * n))
    stat = bn * (np.log(np.mean(rvs)) - np.mean(np.log(rvs)))
    cdf_val = stats.distributions.chi2.cdf(stat, n - 1)
    pval = 2 * np.minimum(cdf_val, 1 - cdf_val)
    return pval


def holm_bonferroni(pvals):
    sorted_pvals = pvals.sort_values()
    notna = np.sum(~np.isnan(sorted_pvals))
    correction = notna - np.arange(len(sorted_pvals))
    correction = pd.Series(data=correction, index=sorted_pvals.index)
    return np.minimum(pvals * correction.loc[pvals.index], 1)


def calc_tct(
    path_to_real,  # transactions.csv.zip
    path_to_synth,  # any synthetic csv
    synth_seq_len,  # len of sequences to recover correct user_id
):
    df_synth = (
        pd.read_csv(path_to_synth)
        .assign(user_id=lambda df: np.arange(len(df)) // synth_seq_len)
        .assign(days_since_first_tx=lambda df: df.groupby("user_id").time_diff_days.cumsum())
    )
    df_real = pd.read_csv(
        path_to_real,
        parse_dates=["transaction_dttm"],
        index_col="user_id",
        header=0,
    )

    mcc_randomness_real = (
        df_real
        .set_index(["user_id", "mcc_code", "currency_rk"])
        .groupby(["user_id", "mcc_code", "currency_rk"])
        .days_since_first_tx.diff()
        .dropna()
        .groupby(["mcc_code"])
        .agg(test_for_exp)
        .fillna(1.0)
        .pipe(holm_bonferroni)
        .pipe(lambda s: s < CONF_LEVEL)
        .rename("real")
    )
    mcc_randomness_synth = (
        df_synth
        .set_index(["user_id", "mcc_code", "currency_rk"])
        .groupby(["user_id", "mcc_code", "currency_rk"])
        .days_since_first_tx.diff()
        .dropna()
        .groupby(["mcc_code"])
        .agg(test_for_exp)
        .fillna(1.0)
        .pipe(holm_bonferroni)
        .pipe(lambda s: s < CONF_LEVEL)
        .rename(f"synth")
    )

    return (
        pd.concat(
            (mcc_randomness_real, mcc_randomness_synth),
            keys=["real", "synth"],
            axis=1,
        ).pipe(lambda df: df.eval("(real == synth).sum()") / len(df))
    )

if __name__ == "__main__":
    args = parse_args()
    print(calc_tct(args.orig, args.data, args.n_rows))