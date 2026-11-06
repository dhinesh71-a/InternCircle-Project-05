# InternCircle Project 05 — Streamlit Titanic Survival Predictor

An educational Streamlit app that loads a Scikit-Learn Decision Tree model and estimates a Titanic passenger's survival probability.

## Files
- `app.py` — Streamlit user interface
- `requirements.txt` — deployment dependencies
- `model.joblib` — trained model file (create this by running the Project 05 training notebook)

## 1. Create `model.joblib`
1. Open the Project 05 training notebook in Google Colab.
2. Run all cells.
3. Download `model.joblib` when prompted.
4. Place `model.joblib` in this same folder, beside `app.py`.

The model must use these features in exactly this order: `pclass`, `sex`, `age`, `fare`.
Sex encoding: female = 1, male = 0.

## 2. Test locally (optional)
Install Python 3.10+ and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Publish on GitHub
1. Create a public GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `model.joblib`.
3. Do not upload private data or personal information.

## 4. Deploy for free with Streamlit Community Cloud
1. Sign in at https://share.streamlit.io/ using GitHub.
2. Choose **Create app**.
3. Select your repository, branch, and `app.py` as the main file.
4. Deploy and wait for the app URL.
5. Open the public URL on your phone and test a prediction.

## Educational note
This app is for learning only. The Titanic dataset is historical and limited. Predictions are not advice or a statement of what would certainly have happened.
