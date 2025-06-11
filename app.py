import streamlit as st
import joblib
import numpy as np

# Load trained model and vectorizer from files
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Title for your web app
st.title("MessageGuard: SMS Spam Detection")

st.write("""
This app classifies SMS messages as Spam or Ham using Machine Learning.
""")

# Text input box
message = st.text_area("Enter your SMS message:", "")

# When user clicks 'Predict'
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform user message into feature vector
        vect_message = vectorizer.transform([message])
        prediction = model.predict(vect_message)[0]
        prob = model.predict_proba(vect_message)[0]
        confidence = np.max(prob) * 100

        # Show result
        if prediction == 1:
            st.error(f"Prediction: Potential SPAM 🚫")
        else:
            st.success(f"Prediction: Genuine ✅")

        st.write(f"Confidence: {confidence:.2f}%")
