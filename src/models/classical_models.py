"""Sparse-compatible models selected for calibrated multiclass text classification."""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
class ClassicalModels:
 def __init__(self,medium_weight=1.0):
  weights={0:1.,1:float(medium_weight),2:1.}
  self.models={'logistic_regression':LogisticRegression(C=2,max_iter=2500,class_weight=weights,solver='saga',n_jobs=-1),'linear_svm':CalibratedClassifierCV(LinearSVC(C=1.0,class_weight=weights,max_iter=10000),method='sigmoid',cv=3)}
 def train(self,X,y):
  for model in self.models.values(): model.fit(X,y)
  return self.models
