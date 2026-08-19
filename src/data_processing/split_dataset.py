"""Stratified splitting with group isolation where an entire usable group column exists."""
import pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from src.config import settings,TRAIN_FILE,VAL_FILE,TEST_FILE
def split_dataset(df,label_column='risk_label',group_column='user_id',seed=settings.random_seed):
 if label_column not in df: raise ValueError(f'Missing target column: {label_column}')
 ratio=settings.validation_size+settings.test_size
 # Group splitting prevents user leakage but only if group IDs and stratification are both viable.
 if group_column in df and df[group_column].notna().all() and df[group_column].nunique()>1:
  first=GroupShuffleSplit(n_splits=1,test_size=ratio,random_state=seed); ti,hi=next(first.split(df,groups=df[group_column])); train,holdout=df.iloc[ti],df.iloc[hi]
  second=GroupShuffleSplit(n_splits=1,test_size=settings.test_size/ratio,random_state=seed); vi,xi=next(second.split(holdout,groups=holdout[group_column])); return train.copy(),holdout.iloc[vi].copy(),holdout.iloc[xi].copy()
 train,holdout=train_test_split(df,test_size=ratio,stratify=df[label_column],random_state=seed); val,test=train_test_split(holdout,test_size=settings.test_size/ratio,stratify=holdout[label_column],random_state=seed); return train.copy(),val.copy(),test.copy()
def save_splits(train,validation,test):
 for frame,path in ((train,TRAIN_FILE),(validation,VAL_FILE),(test,TEST_FILE)):
  path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)
