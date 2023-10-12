#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import streamlit as st

from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[32]:


df = pd.read_csv('./data/ytcb.csv')


# In[34]:


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


# In[37]:


df['Clickbait'] = df['Title'].apply(label_clickbait)


# In[2]:


model = tf.keras.models.load_model('./model.keras')
model.load_weights('./balancing.h5')


# In[30]:


model.summary()


# In[47]:


x = df['Title'].values
labels = df['Clickbait'].values


# In[48]:


maxlen = 500
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x)


# In[49]:


x_train, x_test, y_train, y_test = train_test_split(x, labels)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# In[50]:


loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


# In[51]:


new_model = tf.keras.models.load_model('./model')
new_model.summary()


# In[52]:


loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


# In[55]:


st.title('Clickbait Detection App')
user_input = st.text_input('Enter Video Title')


# In[56]:


if st.button('Detect Clickbait'):
    test = [user_input]
    token_text = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=maxlen)
    preds = [round(i[0]) for i in model.predict(token_text)]
    for (text, pred) in zip(test, preds):
        label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'
        print("{} - {}".format(text, label))
        st.write(f'The video title "{user_input}" is most likely: {label}')


# In[ ]:




