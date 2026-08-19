import re
class TextCleaner:
 """Social-text normalisation retaining punctuation/emojis as affective evidence."""
 def __init__(self,lowercase=True): self.lowercase=lowercase
 def clean(self,text):
  if text is None:return ''
  text=str(text).lower() if self.lowercase else str(text); text=re.sub(r'<[^>]+>',' ',text); text=re.sub(r'https?://\S+|www\.\S+',' <url> ',text); text=re.sub(r'@\w+',' <user> ',text); text=re.sub(r'#(\w+)',r' \1 ',text); text=re.sub(r'(.)\1{3,}',r'\1\1\1',text)
  return re.sub(r'\s+',' ',text).strip()
 def clean_series(self,texts): return texts.fillna('').map(self.clean)
