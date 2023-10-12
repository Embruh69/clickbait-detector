#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import tensorflow as tf
import re
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


df = pd.read_csv('./ytcb.csv')


def label_clickbait(title):
    clickbait_keywords = ["shocking", "you won't believe", "amazing", "powerful", "unbelievable", "exclusive",'top','the best','rare', 'best', 'insane', 'crazy', 'challenge']
    forbidden_patterns = [
        r'\d+\s+foods\s+you\s+need\s+to\s+eat\s+before\s+you\s+die',
        r'\d+\s+food\s+you\s+need\s+to\s+eat\s+in\s+your\s+lifetime',
    ]
    
    for keyword in clickbait_keywords:
        if keyword in title.lower():
            return 1

    for pattern in forbidden_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return 1
            
    if title.isupper():
        return 1
        
    return 0


df['Clickbait'] = df['Title'].apply(label_clickbait)


model = tf.keras.models.load_model('./model.keras')
model.load_weights('./balancing.h5')


model.summary()


x = df['Title'].values
labels = df['Clickbait'].values


maxlen = 500
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x)


st.title('Clickbait Detection App')
user_input = st.text_input('Enter Video Title')


if st.button('Detect Clickbait'):
    test = [user_input]
    token_text = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=maxlen)
    preds = [round(i[0]) for i in model.predict(token_text)]
    for (text, pred) in zip(test, preds):
        label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'
        print("{} - {}".format(text, label))
        st.write(f'The video title "{user_input}" is most likely: {label}')