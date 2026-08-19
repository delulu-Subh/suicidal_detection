from __future__ import annotations
from functools import lru_cache
import joblib
from scipy.sparse import hstack,csr_matrix
from src.config import ARTIFACT_PATH,ID_TO_LABEL
@lru_cache(maxsize=1)
def load_artifacts(path=str(ARTIFACT_PATH)):
 if not ARTIFACT_PATH.exists(): raise FileNotFoundError('No trained model artifact found. Run `python -m src.train` first.')
 return joblib.load(path)
def _contributions(artifact,X,predicted,limit=8):
 model=artifact['model'];base=getattr(model,'calibrated_classifiers_',None)
 if not base:return []
 estimator=base[0].estimator;coef=getattr(estimator,'coef_',None)
 if coef is None:return []
 # Calibrated LinearSVC has one coefficient row per internal class.
 names=(artifact['tfidf'].feature_names() if hasattr(artifact['tfidf'],'feature_names') else list(artifact['tfidf'].get_feature_names_out()))+['positive_rate','negative_rate','polarity','exclamation_count','question_count','text_length'];row=coef[predicted] if coef.shape[0]>1 else coef[0];scores=X.toarray()[0]*row;idx=scores.argsort()[-limit:][::-1]
 return [{'feature':names[i],'contribution':float(scores[i])} for i in idx if scores[i]>0]
def predict_text(text:str)->dict:
 if not isinstance(text,str) or not text.strip(): raise ValueError('text must be a non-empty string.')
 a=load_artifacts();cleaned=a['cleaner'].clean(text);X=hstack([a['tfidf'].transform([cleaned]),csr_matrix(a['sentiment'].transform([cleaned]))],format='csr');probs=a['model'].predict_proba(X)[0];classes=a['model'].classes_;i=int(probs.argmax());encoded=int(classes[i]);return {'predicted_risk':ID_TO_LABEL[encoded],'confidence':float(probs[i]),'probabilities':{ID_TO_LABEL[int(c)]:float(p) for c,p in zip(classes,probs)},'contributing_features':_contributions(a,X,encoded),'model_name':a.get('model_name','classical_model'),'explanation':'Features listed are model contributions to this research classification; they do not establish intent or a clinical diagnosis.'}
