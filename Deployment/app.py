import streamlit as st
import joblib
import numpy as np


model = joblib.load('Deployment/spam_model.pkl')
vectorizer = joblib.load('Deployment/vectorizer.pkl')


st.title("SMS Spam Detection")

st.write("""
This app classifies SMS messages as Spam or Ham using Machine Learning.
""")


message = st.text_area("Enter your SMS message:", "")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        
        vect_message = vectorizer.transform([message])
        prediction = model.predict(vect_message)[0]
        prob = model.predict_proba(vect_message)[0]
        confidence = np.max(prob) * 100

       
        if prediction == 1:
            st.error(f"Prediction: Potential SPAM 🚫")
        else:
            st.success(f"Prediction: Genuine ✅")

        st.write(f"Confidence: {confidence:.2f}%")
