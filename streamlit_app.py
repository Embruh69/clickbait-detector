#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[7]:


model = tf.keras.models.load_model('./model.keras')


# In[ ]:


st.title('Clickbait Detection App')
user_input = st.text_input('Enter Video Title')


# In[ ]:
maxlen = 500
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([user_input])


if st.button('Detect Clickbait'):
    test = [user_input]
    token_text = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=maxlen)
    preds = [round(i[0]) for i in model.predict(token_text)]
    for (text, pred) in zip(test, preds):
        label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'
        st.write(f'The video title "{user_input}" is most likely: {label}')

