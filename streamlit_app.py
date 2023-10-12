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


if st.button('Detect Clickbait'):
    if prediction == 1:
        st.write('The video title is clickbait.')
    else:
        st.write('The video title is not clickbait.')

