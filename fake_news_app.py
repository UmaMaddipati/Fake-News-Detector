import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK stopwords 
nltk.download('stopwords')

# Load and preprocess dataset
@st.cache_resource
def load_data_and_train():
    # Load CSV files 
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    df['clean_text'] = df['text'].apply(clean_text)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    # Train the model
    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer, clean_text

# Load model and vectorizer
model, vectorizer, clean_text = load_data_and_train()


st.title("📰 Fake News Detector")
st.write("Enter a news article and the model will predict if it's real or fake.")

user_input = st.text_area("📝 Enter News Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("🚨 This news is **FAKE**.")
