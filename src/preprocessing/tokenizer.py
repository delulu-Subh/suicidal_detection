from src.config import settings
class BertTokenizerWrapper:
 def __init__(self,model_name=settings.bert_model_name):
  try: from transformers import AutoTokenizer
  except ImportError as e: raise ImportError('Install transformers to use BERT features.') from e
  self.tokenizer=AutoTokenizer.from_pretrained(model_name)
 def encode(self,texts): return self.tokenizer(texts,padding=True,truncation=True,max_length=settings.max_sequence_length,return_tensors='pt')
