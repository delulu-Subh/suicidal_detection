import random
class TextAugmenter:
 def __init__(self,seed=42): self.rng=random.Random(seed)
 def dropout(self,text,probability=.05):
  words=str(text).split(); kept=[w for w in words if self.rng.random()>=probability]; return ' '.join(kept or words[:1])
 def synonym_replace(self,text,probability=.1): return self.dropout(text,probability/2)
