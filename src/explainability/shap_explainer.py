"""Sparse linear SHAP explanations on a small representative sample only."""
def explain_linear_model(artifact,texts,max_samples=25):
 import shap
 from scipy.sparse import hstack,csr_matrix
 cleaned=[artifact['cleaner'].clean(t) for t in list(texts)[:max_samples]];X=hstack([artifact['tfidf'].transform(cleaned),csr_matrix(artifact['sentiment'].transform(cleaned))],format='csr')
 # Calibrated LinearSVC exposes fitted base estimators; explain the first for a representative linear explanation.
 estimator=artifact['model'].calibrated_classifiers_[0].estimator;return shap.LinearExplainer(estimator,X),X,(artifact['tfidf'].feature_names() if hasattr(artifact['tfidf'],'feature_names') else list(artifact['tfidf'].get_feature_names_out()))+['positive_rate','negative_rate','polarity','exclamation_count','question_count','text_length']
