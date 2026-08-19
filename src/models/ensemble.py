import numpy as np
class EnsembleClassifier:
 def __init__(self,models,weights=None): self.models=models; self.weights=np.asarray(weights or [1]*len(models),dtype=float); self.weights/=self.weights.sum()
 def predict_proba(self,X): return sum(w*m.predict_proba(X) for w,m in zip(self.weights,self.models))
 def predict(self,X): return self.predict_proba(X).argmax(axis=1)
