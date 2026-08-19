"""LIME explanation for the saved raw-text inference interface."""
def explain_prediction(text,artifact,output_html=None,num_features=10):
 from lime.lime_text import LimeTextExplainer
 from scipy.sparse import hstack,csr_matrix
 labels=artifact['labels'];class_names=[labels[i] for i in sorted(labels)]
 def classifier(texts):
  cleaned=[artifact['cleaner'].clean(t) for t in texts];X=hstack([artifact['tfidf'].transform(cleaned),csr_matrix(artifact['sentiment'].transform(cleaned))],format='csr');return artifact['model'].predict_proba(X)
 exp=LimeTextExplainer(class_names=class_names).explain_instance(text,classifier,num_features=num_features)
 if output_html: exp.save_to_file(str(output_html))
 return exp
