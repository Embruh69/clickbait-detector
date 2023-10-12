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


if st.button('Detect Clickbait'):
    # Process user input
    token_text = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=maxlen)
    
    # Make prediction
    prediction = model.predict(token_text)
    pred_label = 'Clickbait' if round(prediction[0][0]) == 1 else 'Not Clickbait'
    
    # Display the result
    st.write(f'The video title "{user_input}" is most likely: {pred_label}')

