# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import streamlit as st
import matplotlib.pyplot as plt

max_len = 100

# Load the Twitter dataset
tweet = pd.read_csv('Python\ML\Project\Tweets.csv')
tweet = tweet.dropna(subset=['text'])

# Tokenizer setup
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tweet['text'].astype(str))
word_index = tokenizer.word_index

# Load the pre-trained model
model = load_model('Python\ML\Project\TweeterSentimentAnalysis.h5')

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    words = text.split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# Streamlit app
st.title('✈️ Airline Tweet Sentiment Analysis')
st.write('Enter a Tweet to classify its sentiment (Positive or Negative).')

# User input
user_input = st.text_area('✍️ Enter your Tweet here:')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)[0][0] 
    sentiment = 'Positive' if prediction < 0.5 else 'Negative'
    confidence = 1 - prediction if sentiment == 'Positive' else prediction

    st.success(f'Sentiment: {sentiment}')
    st.info(f'Confidence Score: {confidence:.2f}')
    st.subheader("📈 Sentiment Confidence")
    labels = ['Positive', 'Negative']
    scores = [1 - prediction, prediction]

    fig, ax = plt.subplots()
    ax.barh(labels, scores, color=['green', 'red'])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Sentiment Prediction Confidence")
    st.pyplot(fig)

else:
    st.write('Please enter a Tweet.')
