from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import LABELS
def load_dataset(path:Path|str,text_column='text',label_column='risk_label')->pd.DataFrame:
 df=pd.read_csv(path)
 if text_column not in df or label_column not in df: raise ValueError(f'{path} must contain {text_column!r} and {label_column!r}')
 df=df.dropna(subset=[text_column,label_column]).copy(); df[text_column]=df[text_column].astype(str); return df
def encode_labels(labels):
 unknown=set(labels)-set(LABELS)
 if unknown: raise ValueError(f'Unsupported risk labels: {sorted(unknown)}')
 return labels.map(LABELS).astype(int)
