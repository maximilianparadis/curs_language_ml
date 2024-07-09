#!/usr/bin/env python
# coding: utf-8

# # Sessió 1 deep learning
# 
# conatenació de regressions logistiques
# 
# - crear classificacio a través d'aquestes reg.

# In[1]:


import pandas as pd
import numpy as np
import keras 
import tensorflow as tf


# In[35]:


# importacio de dades

texts = ['per a tu i per a mi', 'per a tothom']

# creem tokens

tokenizer = keras.layers.TextVectorization(output_mode = 'count')


# In[37]:


tokenizer.adapt(texts)
tokenizer(texts)


# # - Natejar el text

# In[42]:


import spacy


# In[43]:


nlp = spacy.load('ca_core_news_sm')


# In[48]:


doc = nlp("l'idescat és l'organ estadístic de la Generalitat de Catalunya")

for token in doc:
    print(token.text, token.pos_, token.lemma_, token.is_stop)


# # - exercici:

# In[2]:


from datasets import load_dataset


# In[61]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


dataset = load_dataset('projecte-aina/tecla')


# In[54]:


# variable dependent


noticies = dataset['train']['label1']
noticies

dicotomiques_noticies = [1 if noticia == 1 else 0 for noticia in noticies]
dicotomiques_noticies_train = dicotomiques_noticies[0:30]
dicotomiques_noticies_train


# In[55]:


# variable independent

text_noticies_train = dataset['train']['text'][0:30]  # si agafem tota la base de dades peta
text_noticies_train
tokenizer = keras.layers.TextVectorization(output_mode = 'count')
tokenizer.adapt(text_noticies_train)
text_noticies_train_var = tokenizer(text_noticies_train)
text_noticies_train_var


# In[56]:


model = LogisticRegression().fit(text_noticies_train_var, dicotomiques_noticies_train)
# model.predict(text_noticies_train_var)
# model.score(text_noticies_train_var, dicotomiques_noticies_train)


# In[57]:


# test , no fem adapt perque el vocabulari de "test" i "train" és diferent

text_noticies_test = dataset['test']['text'][0:30]
tokenizer = keras.layers.TextVectorization(output_mode = 'count')
tokenizer.adapt(text_noticies_train) # adaptem amb el vocabulari anterior
text_noticies_test_var = tokenizer(text_noticies_test)
# text_noticies_test_var


# In[63]:


prediccions = model.predict(text_noticies_test_var)
accuracy = accuracy_score(dicotomiques_noticies_train, prediccions)
accuracy

