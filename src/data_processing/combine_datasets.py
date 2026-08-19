"""Explicit schema normalisation for supplied datasets; ambiguous Twitter labels stay excluded."""
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR,PROCESSED_DATA_DIR
SPECS={'Combined Data.csv':('statement','status',{'Normal':'LOW','Anxiety':'MEDIUM','Stress':'MEDIUM','Depression':'MEDIUM','Bipolar':'MEDIUM','Personality disorder':'MEDIUM','Suicidal':'HIGH'}),'reddit_depression_suicidewatch.csv':('text','label',{'depression':'MEDIUM','SuicideWatch':'HIGH'}),'Relabelled_Cleaned_Dataset.csv':('text','label',{0:'LOW',1:'MEDIUM',2:'HIGH'}),'Suicide_Detection.csv':('text','class',{'non-suicide':'LOW','suicide':'HIGH'})}
def combine_datasets(raw_dir:Path=RAW_DATA_DIR,output_path:Path|None=None)->pd.DataFrame:
 frames=[]
 for name,(text,label,mapping) in SPECS.items():
  path=raw_dir/name
  if not path.exists(): continue
  df=pd.read_csv(path,usecols=[text,label],low_memory=False)
  out=pd.DataFrame({'text':df[text],'original_label':df[label]}); out['risk_label']=out.original_label.map(mapping); out['source_dataset']=name
  frames.append(out.dropna(subset=['text','risk_label']))
 if not frames: raise FileNotFoundError(f'No supported dataset was found in {raw_dir}')
 result=pd.concat(frames,ignore_index=True); result.text=result.text.astype(str); result=result[result.text.str.strip().ne('')].drop_duplicates(['text','risk_label'])
 if output_path: output_path.parent.mkdir(parents=True,exist_ok=True); result.to_csv(output_path,index=False)
 return result
if __name__=='__main__': print(combine_datasets(output_path=PROCESSED_DATA_DIR/'cleaned_data.csv').risk_label.value_counts())
