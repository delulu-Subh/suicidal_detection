# Detection of Suicidal Tendency by Analyzing Social Network Activities

A reproducible research prototype that classifies supplied public social-network text into **LOW**, **MEDIUM**, and **HIGH** observed-risk patterns. It is **not a clinical diagnostic system** and must not be used as a substitute for professional assessment or emergency support.

## Setup (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python -m src.train
# Optional transformer experiment (requires Hugging Face model access)
python -c "from src.train import train_bert; print(train_bert())"
uvicorn app.api:app --reload
streamlit run dashboard/streamlit_app.py
```

## Method

`combine_datasets.py` normalizes explicitly documented source labels while retaining `original_label` and `source_dataset`. The compatible training data has `text` and `risk_label`; the Twitter activity dataset is deliberately excluded because its local binary label semantics are undocumented. Therefore behavioral metadata is not fabricated.

The split is stratified 70/15/15. A group-aware path is automatically used when a complete `user_id` field is genuinely available. TF-IDF is fit only on the training partition. Validation selects the final model by Macro F1 with MEDIUM F1 as a tie-breaker; the test partition is used once for the selected final model.

Outputs are saved under `models/`, `reports/metrics/`, and `reports/figures/`. The dashboard and FastAPI load these artifacts and never retrain.

## Limitations and ethics

Public data may contain sampling, annotation, demographic, language, cultural, and domain-shift bias. Class imbalance can reduce MEDIUM detection. False positives and false negatives are possible. Inputs should not be persistently logged; no identifiers are returned. Results describe contributing model features, not intent, diagnosis, or a person?s mental state. Human oversight and appropriate professional processes remain essential.
