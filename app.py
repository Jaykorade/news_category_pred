import streamlit as st
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer


@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("sentiment_model.pkl")  
    vectorizer = joblib.load("vectorizer.pkl") 
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


st.title("News category prediction")

st.markdown("This app predicts new category.")


user_review = st.text_area("Enter your news here:")

if st.button("Predict news category"):
    if user_review.strip():

        review_vectorized = vectorizer.transform([user_review])
        sentiment = model.predict(review_vectorized)[0]
        if sentiment == 1:
            sentiment_label = "Crime"
        elif sentiment == 2:
            sentiment_label = "Stock"
        elif sentiment == 3:
            sentiment_label = "Politics"
        elif sentiment == 4:
            sentiment_label = "Tech"



        st.subheader(f"Predicted news: {sentiment_label}")
    else:
        st.warning("Please enter a news.")

st.markdown("---")
st.markdown("### About this App")
st.write("The model is trained on the news dataset using a machine learning classifier. It uses text vectorization techniques to analyze the news and predict news category.")