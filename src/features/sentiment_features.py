import re
import numpy as np
import pandas as pd
POS={'happy','hope','love','better','good','safe','help'}; NEG={'sad','pain','hurt','hate','alone','worthless','die','death','suicide','kill'}
class SentimentExtractor:
 """Transparent lexicon counts; not a clinical emotion assessment."""
 def analyze(self,text):
  words=re.findall(r"[a-z']+",str(text).lower()); n=max(len(words),1); pos=sum(w in POS for w in words); neg=sum(w in NEG for w in words)
  return {'positive_rate':pos/n,'negative_rate':neg/n,'polarity':(pos-neg)/n,'exclamation_count':str(text).count('!'),'question_count':str(text).count('?'),'text_length':len(words)}
 def batch_analyze(self,texts,**_): return [self.analyze(t) for t in texts]
 def transform(self,texts): return pd.DataFrame(self.batch_analyze(texts)).to_numpy(dtype=float)
