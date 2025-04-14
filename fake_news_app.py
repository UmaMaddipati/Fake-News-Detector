

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

nltk.download('stopwords')

# 🧹 Step 2: Preprocess Text
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

# 📁 Step 3: Load Model & Vectorizer
@st.cache(allow_output_mutation=True)
def load_model():
    # Load pre-trained model and vectorizer (can be done using pickle or joblib)
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['clean_text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = load_model()

# 📱 Step 4: Streamlit UI
st.title("Fake News Detection")
st.write("Enter the text of a news article to check if it's real or fake.")

news_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    if news_input:
        cleaned_text = clean_text(news_input)
        vectorized_input = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_input)

        if prediction == 1:
            st.success("✅ This news is REAL!")
        else:
            st.error("❌ This news is FAKE!")
    else:
        st.warning("Please enter a news article to predict.")
