
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩"
)

st.title("📩 SMS Spam Detection System")
st.write("Detect spam messages using Machine Learning.")

# Load the trained model
model_path = Path(__file__).parent / "spam_model.joblib"

if not model_path.exists():
    st.error("Model not found. Run python train.py first.")
    st.stop()

model = joblib.load(model_path)

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type a message here..."
)

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("🚨 This message is classified as SPAM!")
        else:
            st.success("✅ This message is classified as NOT SPAM.")

        probabilities = model.predict_proba([message])[0]
        spam_index = list(model.classes_).index("spam")
        spam_score = probabilities[spam_index]

        st.write(f"Model spam score: {spam_score:.1%}")
        st.caption(
            "This is a model-generated score, "
            "not a calibrated probability or guarantee."
        )
