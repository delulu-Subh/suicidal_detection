"""Reproducible, leakage-controlled classical experiment and final-model training."""
from __future__ import annotations
import json,random
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from scipy.sparse import hstack,csr_matrix
from src.config import *
from src.data_processing.combine_datasets import combine_datasets,SPECS
from src.data_processing.split_dataset import split_dataset,save_splits
from src.preprocessing.cleaner import TextCleaner
from src.features.tfidf_features import TFIDFFeatureExtractor
from src.features.sentiment_features import SentimentExtractor
from src.dataset import encode_labels
from src.models.classical_models import ClassicalModels
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.evaluation.metrics import evaluate_predictions,multiclass_curves
from src.evaluation.visualization import save_confusion_matrix,save_class_distribution,save_curves,save_comparison
def _features(extractor,sentiment,texts,fit=False):
 matrix=extractor.fit_transform(texts) if fit else extractor.transform(texts); return hstack([matrix,csr_matrix(sentiment.transform(texts))],format='csr')
def _sample(frame):
 if not settings.max_training_rows or len(frame)<=settings.max_training_rows:return frame
 each=max(1,settings.max_training_rows//frame.risk_label.nunique());return frame.groupby('risk_label',group_keys=False).sample(n=each,random_state=settings.random_seed)
def _row(name,feature,weight,metrics):
 p=metrics['per_class']['1'];return {'model':name,'feature_representation':feature,'medium_weight':weight,'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],'weighted_f1':metrics['weighted_f1'],'roc_auc':metrics.get('roc_auc_ovr_macro'),'pr_auc':metrics.get('pr_auc_macro'),'medium_recall':p['recall'],'medium_f1':p['f1-score']}
def train_classical():
 random.seed(settings.random_seed);np.random.seed(settings.random_seed); METRICS_DIR.mkdir(parents=True,exist_ok=True);MODEL_DIR.mkdir(parents=True,exist_ok=True)
 raw=combine_datasets(output_path=PROCESSED_DATA_DIR/'cleaned_data.csv');summary={'records_after_normalization':len(raw),'class_counts':raw.risk_label.value_counts().to_dict(),'missing_text':int(raw.text.isna().sum()),'duplicate_text_label_removed':'yes, before splitting','text_column':'text','target_column':'risk_label','behavioral_features':'Not used: compatible normalized sources do not include user/activity metadata. The Twitter source is excluded because local labels 0/1 lack documented risk semantics.','split_policy':'stratified 70/15/15; group-aware only when a complete user_id column is present.'};(METRICS_DIR/'dataset_summary.json').write_text(json.dumps(summary,indent=2));(METRICS_DIR/'label_mapping.json').write_text(json.dumps({'internal_mapping':LABELS,'source_mappings':{k:{str(a):b for a,b in spec[2].items()} for k,spec in SPECS.items()}},indent=2))
 cleaner=TextCleaner();raw.text=cleaner.clean_series(raw.text);train,val,test=split_dataset(raw);save_splits(train,val,test);save_class_distribution(raw.risk_label.value_counts()); train=_sample(train);ytrain,yval,ytest=(encode_labels(x.risk_label) for x in (train,val,test)); sentiment=SentimentExtractor();experiments=[('word_tfidf',False,1.0),('word_tfidf_medium_weighted',False,1.5)] + ([('word_char_tfidf_medium_weighted',True,1.5)] if __import__('os').getenv('RISK_INCLUDE_CHAR_EXPERIMENT','false').lower()=='true' else []);records=[];candidate=None
 for title,chars,weight in experiments:
  extractor=TFIDFFeatureExtractor(include_char=chars,char_max_features=12000);Xtrain=_features(extractor,sentiment,train.text,True);Xval=_features(extractor,sentiment,val.text);model=LinearSVC(C=1.0,class_weight={0:1.,1:weight,2:1.},max_iter=10000);model.fit(Xtrain,ytrain);score=model.decision_function(Xval);prob=np.exp(score-score.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);metrics=evaluate_predictions(yval,model.predict(Xval),prob);record=_row(f'linear_svm_screen:{title}',title,weight,metrics);records.append(record)
  if candidate is None or (record['macro_f1'],record['medium_f1'])>(candidate[0]['macro_f1'],candidate[0]['medium_f1']):candidate=(record,extractor)
 chosen,extractor=candidate;Xtrain=_features(extractor,sentiment,train.text,True);Xval=_features(extractor,sentiment,val.text);models=ClassicalModels(chosen['medium_weight']).models;fitted={}
 for name,model in models.items(): model.fit(Xtrain,ytrain);metrics=evaluate_predictions(yval,model.predict(Xval),model.predict_proba(Xval));records.append(_row(f'{name}:selected_features',chosen['feature_representation'],chosen['medium_weight'],metrics));fitted[name]=model
 selected=max(records[-2:],key=lambda r:(r['macro_f1'],r['medium_f1']));final_model=fitted[selected['model'].split(':')[0]];Xtest=_features(extractor,sentiment,test.text);probs=final_model.predict_proba(Xtest);test_metrics=evaluate_predictions(ytest,final_model.predict(Xtest),probs);curves=multiclass_curves(ytest,probs);save_confusion_matrix(test_metrics['confusion_matrix'],name='final_test_confusion_matrix');save_curves(curves,'final_test');pd.DataFrame(records).to_csv(METRICS_DIR/'model_comparison.csv',index=False);(METRICS_DIR/'model_comparison.json').write_text(json.dumps({'validation_experiments':records,'final_selection':selected,'final_test':test_metrics,'generated_at':datetime.now(timezone.utc).isoformat()},indent=2));(METRICS_DIR/'classification_reports.json').write_text(json.dumps({'final_test':test_metrics},indent=2));pd.DataFrame(test_metrics['confusion_matrix'],index=['LOW','MEDIUM','HIGH'],columns=['LOW','MEDIUM','HIGH']).to_csv(METRICS_DIR/'confusion_matrix.csv');save_comparison(records)
 artifact={'model':final_model,'tfidf':extractor,'sentiment':sentiment,'labels':ID_TO_LABEL,'cleaner':cleaner,'model_name':selected['model'],'feature_representation':selected['feature_representation'],'trained_at':datetime.now(timezone.utc).isoformat()};joblib.dump(artifact,ARTIFACT_PATH);METADATA_PATH.write_text(json.dumps({'selected_model':selected,'test':test_metrics,'dataset_summary':summary},indent=2));return {'best_model':selected['model'],'test_metrics':test_metrics,'validation_records':records}

def train_bert(max_rows=None):
 """Train optional DistilBERT on training data and evaluate against validation/test without leakage."""
 from src.models.bert_model import BERTTrainer
 train=pd.read_csv(TRAIN_FILE);val=pd.read_csv(VAL_FILE);test=pd.read_csv(TEST_FILE);cleaner=TextCleaner()
 for frame in (train,val,test): frame.text=cleaner.clean_series(frame.text)
 if max_rows:
  train=train.groupby('risk_label',group_keys=False).sample(n=max_rows//3,random_state=settings.random_seed); val=val.groupby('risk_label',group_keys=False).head(max(30,max_rows//6)); test=test.groupby('risk_label',group_keys=False).head(max(30,max_rows//6))
 ytrain,yval,ytest=(encode_labels(x.risk_label).to_numpy() for x in (train,val,test));weights=np.bincount(ytrain,minlength=3);weights=weights.sum()/(3*np.maximum(weights,1))
 trainer=BERTTrainer();history=trainer.fit(train.text.tolist(),ytrain,val.text.tolist(),yval,class_weights=weights);probs=trainer.predict_proba(test.text.tolist());metrics=evaluate_predictions(ytest,probs.argmax(1),probs);path=MODEL_DIR/'bert_risk_classifier';trainer.save(path);(METRICS_DIR/'bert_metrics.json').write_text(json.dumps({'history':history,'test':metrics},indent=2));return metrics
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--with-bert',action='store_true');parser.add_argument('--bert-smoke',action='store_true');args=parser.parse_args();result=train_classical();
 if args.with_bert or args.bert_smoke: result['bert']=train_bert(900 if args.bert_smoke else None)
 print(json.dumps(result,indent=2))
