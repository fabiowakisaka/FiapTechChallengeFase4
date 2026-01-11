import numpy as np
import pandas as pd

def parse_vol(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", "").upper().strip()
        try:
            if x.endswith("B"):
                return float(x[:-1]) * 1e9
            elif x.endswith("M"):
                return float(x[:-1]) * 1e6
            elif x.endswith("K"):
                return float(x[:-1]) * 1e3
            else:
                return float(x)
        except ValueError:
            return np.nan
    elif isinstance(x, (int, float)):
        return float(x)
    return np.nan


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
