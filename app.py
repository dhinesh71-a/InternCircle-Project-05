import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = Path(__file__).parent / "model.joblib"

@st.cache_resource
def load_model():
    """Load the trained model once and cache it for the app session."""
    return joblib.load(MODEL_PATH)

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Explore a simple machine-learning model that estimates the probability "
    "of survival for a hypothetical Titanic passenger."
)

st.info(
    "Educational demonstration only. The model was trained on historical Titanic "
    "passenger data and should not be used for real-world decisions."
)

with st.form("passenger_inputs"):
    pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=2)
    sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    fare = st.slider("Fare paid", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
    submitted = st.form_submit_button("Predict survival probability")

if submitted:
    try:
        if not MODEL_PATH.exists():
            st.error(
                "The model file (model.joblib) is missing. Run the Project 05 "
                "training notebook, download model.joblib, and add it to this "
                "GitHub repository beside app.py."
            )
            st.stop()

        sex_value = 1 if sex_label == "Female" else 0
        input_row = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=FEATURES
        )

        if input_row.isnull().any().any():
            st.error("Please provide a valid value for every input.")
            st.stop()

        model = load_model()
        probabilities = model.predict_proba(input_row)[0]
        survival_probability = float(probabilities[1])
        prediction = int(model.predict(input_row)[0])

        st.metric("Estimated chance of survival", f"{survival_probability:.1%}")
        if prediction == 1:
            st.success("The model predicts: Survived")
        else:
            st.warning("The model predicts: Did not survive")

        st.caption(
            "This is a model estimate, not a historical certainty. It uses only "
            "ticket class, sex, age and fare."
        )

    except FileNotFoundError:
        st.error("Model file not found. Please add model.joblib to the app folder.")
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Sorry, the input could not be processed: {error}")
    except Exception:
        st.error("The prediction could not be completed. Check that model.joblib "
                 "was created using the same four features in the same order.")
