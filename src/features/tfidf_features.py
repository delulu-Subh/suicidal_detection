"""Sparse word/character TF-IDF extraction fitted solely on training text."""
from pathlib import Path
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import settings,MODEL_DIR
class TFIDFFeatureExtractor:
 def __init__(self,include_char=True,**kwargs):
  common=dict(min_df=kwargs.get('min_df',settings.min_df),max_df=kwargs.get('max_df',settings.max_df),max_features=kwargs.get('max_features',settings.max_features),sublinear_tf=True,strip_accents='unicode',dtype='float32')
  self.word=TfidfVectorizer(analyzer='word',ngram_range=(1,kwargs.get('word_ngram_max',settings.word_ngram_max)),**common); self.char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,kwargs.get('char_ngram_max',settings.char_ngram_max)),max_features=kwargs.get('char_max_features',40000),min_df=common['min_df'],max_df=common['max_df'],sublinear_tf=True,dtype='float32') if include_char else None
 def fit_transform(self,texts):
  word=self.word.fit_transform(texts); return hstack([word,self.char.fit_transform(texts)]).tocsr() if self.char else word
 def transform(self,texts):
  word=self.word.transform(texts); return hstack([word,self.char.transform(texts)]).tocsr() if self.char else word
 def feature_names(self): return list(self.word.get_feature_names_out())+(list(self.char.get_feature_names_out()) if self.char else [])
 def save(self,path=None):
  path=Path(path or MODEL_DIR/'tfidf_vectorizer.joblib'); path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)
