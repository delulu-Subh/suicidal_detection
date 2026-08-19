from __future__ import annotations
import numpy as np
from src.config import settings
class BertEmbeddingExtractor:
 def __init__(self,model_name=settings.bert_model_name):
  from transformers import AutoTokenizer,AutoModel
  import torch
  self.torch=torch;self.tokenizer=AutoTokenizer.from_pretrained(model_name);self.model=AutoModel.from_pretrained(model_name);self.model.eval()
 def extract(self,texts,batch_size=None,device=None):
  device=device or ('cuda' if self.torch.cuda.is_available() else 'cpu');self.model.to(device);out=[];batch_size=batch_size or settings.bert_batch_size
  with self.torch.no_grad():
   for i in range(0,len(texts),batch_size):
    inputs=self.tokenizer(list(texts[i:i+batch_size]),padding=True,truncation=True,max_length=settings.max_sequence_length,return_tensors='pt');inputs={k:v.to(device) for k,v in inputs.items()};out.append(self.model(**inputs).last_hidden_state[:,0].cpu().numpy())
  return np.vstack(out) if out else np.empty((0,self.model.config.hidden_size))
