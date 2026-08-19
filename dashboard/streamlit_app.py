import json
import streamlit as st
from app.inference import predict_text
from src.config import METRICS_DIR,FIGURES_DIR
st.set_page_config(page_title='Research Risk Classifier',page_icon='??',layout='wide')
st.title('Social-content risk classification research prototype')
st.warning('This system is a research prototype for classification of suicide-related risk signals in social-network content. It is not a clinical diagnostic system and should not be used as a substitute for professional assessment.')
text=st.text_area('Text to analyze',height=180,placeholder='Enter social-network content?')
if st.button('Analyze',type='primary'):
 try:
  r=predict_text(text);c1,c2=st.columns(2);c1.metric('Predicted risk level',r['predicted_risk']);c2.metric('Model confidence',f"{r['confidence']:.1%}");st.bar_chart(r['probabilities']);st.caption(r['explanation']);st.dataframe(r['contributing_features'],use_container_width=True)
 except Exception as e:st.error(str(e))
st.divider();st.subheader('Research evaluation artifacts')
for path in sorted(FIGURES_DIR.glob('*.png')): st.image(str(path),caption=path.stem)
comparison=METRICS_DIR/'model_comparison.csv'
if comparison.exists():st.dataframe(comparison.read_text(),use_container_width=True)
