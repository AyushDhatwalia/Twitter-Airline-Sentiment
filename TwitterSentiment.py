# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_len = 100
# Load the Twitter dataset word index
tweet=pd.read_csv('Python\ML\Project\Tweets.csv')
tweet = tweet.dropna(subset=['negativereason'])
# Tokenizer setup
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tweet['negativereason'].astype(str))
word_index = tokenizer.word_index

# Load the pre-trained model with ReLU activation
model = load_model('Python\ML\Project\TweeterSentimentAnalysis.h5')

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    words = text.split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

### Prediction  function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] < 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Step 4: User Input and Prediction
# Example review for prediction
example_review = "This movie was fantastic! The acting was great and the plot was thrilling."

sentiment,score=predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {1-score}')