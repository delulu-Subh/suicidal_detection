"""Generate bounded LIME/SHAP artifacts without persisting input post text."""
from pathlib import Path
import joblib,json
import matplotlib.pyplot as plt
from src.config import ARTIFACT_PATH,TEST_FILE,FIGURES_DIR,METRICS_DIR
from src.explainability.lime_explainer import explain_prediction
from src.explainability.shap_explainer import explain_linear_model
def run():
 a=joblib.load(ARTIFACT_PATH);import pandas as pd;test=pd.read_csv(TEST_FILE);sample=str(test.iloc[0].text);FIGURES_DIR.mkdir(parents=True,exist_ok=True);METRICS_DIR.mkdir(parents=True,exist_ok=True)
 lime=explain_prediction(sample,a,FIGURES_DIR/'lime_example.html');lime_data=[{'feature':x,'weight':float(w)} for x,w in lime.as_list()];(METRICS_DIR/'lime_example.json').write_text(json.dumps({'note':'feature contributions, not evidence of intent','features':lime_data},indent=2))
 explainer,X,names=explain_linear_model(a,test.text,25);values=explainer(X);plt.figure();import shap;shap.plots.bar(values[:,:,0] if len(values.shape)==3 else values,max_display=15,show=False);plt.tight_layout();plt.savefig(FIGURES_DIR/'shap_feature_contributions.png',dpi=160);plt.close();return lime_data
if __name__=='__main__':print(run())
