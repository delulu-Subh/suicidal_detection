from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR
def inspect_directory(data_dir:Path=RAW_DATA_DIR)->dict:
 report={}
 for path in data_dir.glob('*.csv'):
  sample=pd.read_csv(path,nrows=1000,low_memory=False); report[path.name]={'columns':sample.columns.tolist(),'dtypes':{k:str(v) for k,v in sample.dtypes.items()},'nulls_in_sample':sample.isna().sum().to_dict()}
 return report
if __name__=='__main__':
 import json; print(json.dumps(inspect_directory(),indent=2))
