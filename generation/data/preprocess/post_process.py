import argparse
import pandas as pd
import numpy as np
from collections import Counter
import pickle


def extract_ngrams(seq: list[int], n: int = 3) -> list[tuple]:
    L = len(seq)
    if L < n:
        return []
    return [tuple(seq[i : i + n]) for i in range(L - n + 1)]


def select_by_coverage(counter: Counter, coverage: float) -> list[tuple[int, ...]]:
    total = sum(counter.values())
    cum = 0
    selected = []
    for ng, freq in counter.most_common():
        cum += freq
        selected.append(ng)
        if cum / total >= coverage:
            break
    return selected


def merge_ngrams(
    seq: np.ndarray, ngram_map: dict[tuple[int, ...], int], n: int
) -> np.ndarray:
    """
    Однопроходное неперекрывающееся схлопывание n-грамм согласно `ngram_map`.
    """
    out, i = [], 0
    while i < len(seq):
        if i <= len(seq) - n and tuple(seq[i : i + n]) in ngram_map:
            out.append(ngram_map[tuple(seq[i : i + n])])
            i += n
        else:
            out.append(seq[i])
            i += 1
    return np.array(out, dtype=int)


def apply_ngrams(
    df,
    n,
    col:str,
    coverage: float = 0.5,
    new_col="category_merged",
    return_merged: bool = False,
):
    df["ngrams"] = df[col].apply(lambda s: extract_ngrams(s, n=n))
    df["ngram_counts"] = df["ngrams"].apply(Counter)  # Counter per row
    df["unique_ngrams"] = df["ngram_counts"].apply(len)  # сколько разных
    df["total_ngrams"] = df["ngrams"].str.len()  # сколько всего

    ngrams_total_counts = Counter()
    
    for cnt in df["ngram_counts"]:
        ngrams_total_counts.update(cnt)

    selected = select_by_coverage(ngrams_total_counts, coverage=coverage)

    max_token = int(df[col].explode().max())

    ngram_to_token = {ng: idx for idx, ng in enumerate(selected, start=max_token + 1)}

    df[new_col] = df[col].apply(
        lambda seq: merge_ngrams(np.asarray(seq, dtype=int), ngram_to_token, n)
    )

    ngram_to_token = {ng: idx for idx, ng in enumerate(selected, start=max_token + 1)}

    df.drop(columns=["ngrams", "ngram_counts"], inplace=True)

    if return_merged:
        return ngram_to_token
    else:
        return ngram_to_token, df


def main(path_from, path_to, feature_name):
    df = pd.read_parquet(path_from)
    maps = {}
    
    mapping, df = apply_ngrams(
        df, 3, coverage=0.3, col=feature_name, new_col=feature_name
    )
    print(mapping)
    maps[f"{3}-grams"] = mapping
    mapping, df = apply_ngrams(
        df, 2, coverage=0.2, col=feature_name, new_col=feature_name
    )
    print(mapping)
    maps[f"{2}-grams"] = mapping

    with open(path_to, "wb") as file:
        pickle.dump(maps, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and convert MBD dataset to Parquet"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to current dataset"
    )
    parser.add_argument(
        "--feature", type=str, required=True, help='Feature to be ngrammed'
    )
    parser.add_argument(
        "--save-path", type=str, default=False, help="Path to save some modes"
    )

    args = parser.parse_args()

    main(path_from=args.data_path, path_to=args.save_path, feature_name=args.feature)
