
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title="Audience Sentiment Analyzer", page_icon="🎬")
st.title("🎭 Audience Sentiment Analysis for Media")
st.write("Analyze social media posts and understand audience sentiment trends!")

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

option = st.radio("Choose Analysis Type:", ["Single Text Input", "Upload CSV File"])

if option == "Single Text Input":
    user_input = st.text_area("Enter a tweet, comment, or review:")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            cleaned_input = preprocess_text(user_input)
            X = vectorizer.transform([cleaned_input])
            pred = model.predict(X)[0]
            emoji = {"positive":"😊","neutral":"😐","negative":"😠"}
            st.success(f"Predicted Sentiment: {pred.capitalize()} {emoji[pred]}")
        else:
            st.warning("Please enter some text.")
else:
    uploaded_file = st.file_uploader("Upload CSV with a 'Text' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Text" in df.columns:
            df['cleaned_text'] = df['Text'].astype(str).apply(preprocess_text)
            X = vectorizer.transform(df['cleaned_text'])
            df["Predicted_Sentiment"] = model.predict(X)
            st.write("### Sentiment Summary")
            sentiment_counts = df["Predicted_Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)
            st.write("### Sample Results")
            st.dataframe(df.head(10))
        else:
            st.error("CSV must contain a 'Text' column.")

