from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,roc_auc_score,average_precision_score,confusion_matrix,roc_curve,precision_recall_curve
def evaluate_predictions(y_true,y_pred,probabilities=None,labels=(0,1,2)):
 wp,wr,wf,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0); mp,mr,mf,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
 result={'accuracy':float(accuracy_score(y_true,y_pred)),'precision':float(wp),'recall':float(wr),'f1':float(wf),'macro_precision':float(mp),'macro_recall':float(mr),'macro_f1':float(mf),'weighted_precision':float(wp),'weighted_recall':float(wr),'weighted_f1':float(wf),'per_class':classification_report(y_true,y_pred,labels=list(labels),output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y_true,y_pred,labels=list(labels)).tolist()}
 if probabilities is not None:
  try: result['roc_auc_ovr_macro']=float(roc_auc_score(y_true,probabilities,multi_class='ovr',average='macro',labels=list(labels))); result['pr_auc_macro']=float(average_precision_score(np.eye(len(labels))[np.asarray(y_true)],probabilities,average='macro'))
  except ValueError: result['roc_auc_ovr_macro']=None; result['pr_auc_macro']=None
 return result
def multiclass_curves(y_true,probabilities,labels=(0,1,2)):
 out={}; y=np.asarray(y_true)
 for i,label in enumerate(labels):
  binary=(y==label).astype(int); fpr,tpr,_=roc_curve(binary,probabilities[:,i]); precision,recall,_=precision_recall_curve(binary,probabilities[:,i]); out[str(label)]={'fpr':fpr.tolist(),'tpr':tpr.tolist(),'precision':precision.tolist(),'recall':recall.tolist()}
 return out
