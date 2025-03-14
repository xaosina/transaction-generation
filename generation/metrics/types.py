from dataclasses import dataclass
import pandas as pd


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
