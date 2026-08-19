"""Memory-conscious Hugging Face sequence-classification training utilities."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from src.config import settings
class EncodedTextDataset(Dataset):
 def __init__(self,texts,labels,tokenizer): self.encodings=tokenizer(list(texts),truncation=True,padding='max_length',max_length=settings.max_sequence_length);self.labels=list(labels)
 def __len__(self):return len(self.labels)
 def __getitem__(self,i):
  item={k:torch.tensor(v[i]) for k,v in self.encodings.items()};item['labels']=torch.tensor(self.labels[i]);return item
class BERTTrainer:
 def __init__(self,model_name=settings.bert_model_name,num_labels=3,device=None):
  try: from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
  except ImportError as e: raise ImportError('Install transformers to train BERT.') from e
  self.AutoModel=AutoModelForSequenceClassification;self.scheduler_factory=get_linear_schedule_with_warmup;self.tokenizer=AutoTokenizer.from_pretrained(model_name);self.model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels);self.model_name=model_name;self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'));self.model.to(self.device)
 def loader(self,texts,labels,shuffle=False):return DataLoader(EncodedTextDataset(texts,labels,self.tokenizer),batch_size=settings.bert_batch_size,shuffle=shuffle)
 def fit(self,train_texts,train_labels,val_texts,val_labels,class_weights=None):
  loader=self.loader(train_texts,train_labels,True);opt=torch.optim.AdamW(self.model.parameters(),lr=settings.learning_rate);steps=max(1,(len(loader)*settings.bert_epochs+settings.gradient_accumulation-1)//settings.gradient_accumulation);sch=self.scheduler_factory(opt,0,steps);weights=torch.tensor(class_weights,dtype=torch.float,device=self.device) if class_weights is not None else None;scaler=torch.amp.GradScaler('cuda',enabled=self.device.type=='cuda');best=(-1,None);history=[]
  for epoch in range(settings.bert_epochs):
   self.model.train();opt.zero_grad();losses=[]
   for i,batch in enumerate(loader):
    labels=batch.pop('labels').to(self.device);batch={k:v.to(self.device) for k,v in batch.items()}
    with torch.amp.autocast('cuda',enabled=self.device.type=='cuda'):
     logits=self.model(**batch).logits;loss=torch.nn.functional.cross_entropy(logits,labels,weight=weights)/settings.gradient_accumulation
    scaler.scale(loss).backward();losses.append(loss.item()*settings.gradient_accumulation)
    if (i+1)%settings.gradient_accumulation==0 or i+1==len(loader): scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);scaler.step(opt);scaler.update();opt.zero_grad();sch.step()
   probs=self.predict_proba(val_texts);pred=probs.argmax(1);from src.evaluation.metrics import evaluate_predictions;metrics=evaluate_predictions(val_labels,pred,probs);history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics})
   if metrics['macro_f1']>best[0]:best=(metrics['macro_f1'],{k:v.detach().cpu() for k,v in self.model.state_dict().items()})
  if best[1] is not None:self.model.load_state_dict(best[1])
  return history
 def predict_proba(self,texts):
  self.model.eval();out=[]
  with torch.no_grad():
   for batch in self.loader(texts,[0]*len(texts)):
    batch.pop('labels');batch={k:v.to(self.device) for k,v in batch.items()};out.append(self.model(**batch).logits.softmax(1).cpu())
  return torch.cat(out).numpy()
 def save(self,path):
  path=Path(path);path.mkdir(parents=True,exist_ok=True);self.model.save_pretrained(path);self.tokenizer.save_pretrained(path);(path/'training_config.json').write_text(json.dumps({'model_name':self.model_name,'max_length':settings.max_sequence_length}))
 @classmethod
 def load(cls,path,device=None):
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  obj=cls.__new__(cls);obj.tokenizer=AutoTokenizer.from_pretrained(path);obj.model=AutoModelForSequenceClassification.from_pretrained(path);obj.model_name=str(path);obj.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'));obj.model.to(obj.device);return obj
