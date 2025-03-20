from dataclasses import dataclass
import pandas as pd

# TODO: Мб добавить __post_init__ы для assertов
@dataclass
class BinaryData:
    y_true: pd.DataFrame
    y_pred: pd.DataFrame


@dataclass
class CoverageData:
    y_hist: pd.DataFrame
    y_pred: pd.DataFrame


@dataclass
class OnlyPredData:
    y_pred: pd.DataFrame
