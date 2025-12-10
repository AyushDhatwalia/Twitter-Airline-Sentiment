# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model & Tokenizer Setup

max_len = 100

# Load dataset (MAKE SURE THIS FILE EXISTS IN GITHUB REPO)
tweet = pd.read_csv("Tweets.csv")
tweet = tweet.dropna(subset=['text'])

# Tokenizer setup
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tweet['text'].astype(str))
word_index = tokenizer.word_index

# Load trained LSTM model
model = load_model("TweeterSentimentAnalysis.h5")

# Text Preprocessing Function

def preprocess_text(text):
    text = text.lower()
    words = text.split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# Prediction Function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = "Positive" if prediction[0][0] < 0.5 else "Negative"
    return sentiment, prediction[0][0]
    
# STREAMLIT UI

st.title("✈️ Airline Tweet Sentiment Analysis")
st.write("Enter an Airline Tweet to classify its sentiment:")

user_input = st.text_area("Airline Tweet")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a tweet first!")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)

        sentiment = "Positive" if prediction[0][0] < 0.5 else "Negative"

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {1 - prediction[0][0]:.4f}")
else:
    st.info("Please enter a tweet and press Classify.")
