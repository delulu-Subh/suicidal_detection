from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR
def _path(name): FIGURES_DIR.mkdir(parents=True,exist_ok=True); return FIGURES_DIR/f'{name}.png'
def save_confusion_matrix(matrix,labels=('LOW','MEDIUM','HIGH'),name='confusion_matrix'):
 fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(np.asarray(matrix),annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=ax); ax.set(xlabel='Predicted',ylabel='Observed'); fig.tight_layout(); path=_path(name);fig.savefig(path,dpi=160);plt.close(fig);return path
def save_class_distribution(counts,name='class_distribution'):
 ax=counts.plot.bar(color='#3478bf');ax.set(ylabel='Records',xlabel='Risk class');ax.figure.tight_layout();path=_path(name);ax.figure.savefig(path,dpi=160);plt.close(ax.figure);return path
def save_curves(curves,name='test'):
 labels=('LOW','MEDIUM','HIGH');fig,axes=plt.subplots(1,2,figsize=(11,4))
 for i,label in enumerate(labels):
  c=curves[str(i)];axes[0].plot(c['fpr'],c['tpr'],label=label);axes[1].plot(c['recall'],c['precision'],label=label)
 axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='One-vs-rest ROC');axes[1].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest Precision?Recall')
 for ax in axes:ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();path=_path(f'{name}_roc_pr_curves');fig.savefig(path,dpi=160);plt.close(fig);return path
def save_comparison(rows,name='model_comparison'):
 names=[x['model'] for x in rows]; fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(names));ax.bar(x-.18,[x['macro_f1'] for x in rows],.36,label='Macro F1');ax.bar(x+.18,[x['medium_f1'] for x in rows],.36,label='MEDIUM F1');ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylim(0,1);ax.legend();fig.tight_layout();path=_path(name);fig.savefig(path,dpi=160);plt.close(fig);return path
