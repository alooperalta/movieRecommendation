#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
import time
# import asyncio
# import ipywidgets as widgets
import streamlit as st


# In[2]:


url="https://raw.githubusercontent.com/alooperalta/movieRecommendation/master/content/ml-latest-small/movies.csv"
movies=pd.read_csv(url)
movies.head()


# In[78]:


st.title('Movie Recommendation System')


# In[3]:


movies.describe()


# In[4]:


movie_title=movies['title']
Genres = movies['genres'].str.get_dummies(sep='|')
movies = pd.concat([movies, Genres], axis=1)
movies.drop(['genres','(no genres listed)'],axis=1,inplace=True)
movies.head()


# # Data Analysis

# In[5]:


movies.describe()


# In[6]:


# movies.columns


# In[7]:


# Action=movies.loc[movies['Action']==1]
# print(f'Total no. of Action movies: {len(Action)}')
# Action.head()


# # In[8]:


# Adventure=movies.loc[movies['Adventure']==1]
# print(f'Total no. of Adventure movies: {len(Adventure)}')
# Adventure.head()


# # In[9]:


# Animation=movies.loc[movies['Animation']==1]
# print(f'Total no. of Animation movies: {len(Animation)}')
# Animation.head()


# # In[87]:


# Children=movies.loc[movies['Children']==1]
# print(f'Total no. of Children movies: {len(Children)}')
# Children.head()


# # In[86]:


# Comedy=movies.loc[movies['Comedy']==1]
# print(f'Total no. of Comedy movies: {len(Comedy)}')
# Comedy.head()


# # In[12]:


# Crime=movies.loc[movies['Crime']==1]
# print(f'Total no. of Crime movies: {len(Crime)}')
# Crime.head()


# # In[85]:


# Documentary=movies.loc[movies['Documentary']==1]
# print(f'Total no. of Documentary movies: {len(Documentary)}')
# Documentary.head()


# # In[84]:


# Drama=movies.loc[movies['Drama']==1]
# print(f'Total no. of Drama movies: {len(Drama)}')
# Drama.head()


# # In[83]:


# Fantasy=movies.loc[movies['Fantasy']==1]
# print(f'Total no. of Fantasy movies: {len(Fantasy)}')
# Fantasy.head()


# # In[16]:


# Film_Noir=movies.loc[movies['Film-Noir']==1]
# print(f'Total no. of Film-Noir movies: {len(Film_Noir)}')
# Film_Noir.head()


# # In[17]:


# Horror=movies.loc[movies['Horror']==1]
# print(f'Total no. of Horror movies: {len(Horror)}')
# Horror.head()


# # In[18]:


# IMAX=movies.loc[movies['IMAX']==1]
# print(f'Total no. of IMAX movies: {len(IMAX)}')
# IMAX.head()


# # In[19]:


# Musical=movies.loc[movies['Musical']==1]
# print(f'Total no. of Musical movies: {len(Musical)}')
# Musical.head()


# # In[20]:


# Mystery=movies.loc[movies['Mystery']==1]
# print(f'Total no. of Action movies: {len(Mystery)}')
# Mystery.head()


# # In[82]:


# Romance=movies.loc[movies['Romance']==1]
# print(f'Total no. of Romance movies: {len(Romance)}')
# Romance.head()


# # In[22]:


# Sci_fi=movies.loc[movies['Sci-Fi']==1]
# print(f'Total no. of Sci-Fi movies: {len(Sci_fi)}')
# Action.head()


# # In[23]:


# Thriller=movies.loc[movies['Thriller']==1]
# print(f'Total no. of Thriller movies: {len(Thriller)}')
# Thriller.head()


# # In[24]:


# War=movies.loc[movies['War']==1]
# print(f'Total no. of War movies: {len(War)}')
# War.head()


# # In[25]:


# Western=movies.loc[movies['Western']==1]
# print(f'Total no. of Western movies: {len(Western)}')
# Western.head()


# In[ ]:





# # Model Creation

# In[26]:


from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import random


# In[27]:


X=movies.drop('title',axis=1)
n_clusters=170
recommender=KMeans(n_clusters=n_clusters,random_state=1)
labels=recommender.fit_predict(X)


# In[28]:


rndi=dict()
for i in range(n_clusters):
    rndi[i]=[]
for i in range(len(labels)):
    rndi[labels[i]].append(movie_title[i])


# In[29]:


cnt=0
for x in rndi.values():
    if len(x)<6:
        cnt+=1
cnt


# In[88]:


def getRecom(x):
    if(x!="Select an Option"):
        ind=movie_title[movie_title==x].index[0]
        choice=random.sample(rndi[labels[ind]],5)
        st.write('\nFinding Recommendations Similar to', x,'...')
        time.sleep(2)
        st.write('\nTop Matches for',x, 'are:\n' )
        for i in choice:
            if (i==x):
                st.write(random.sample(rndi[labels[ind]],1)[0]) 
            else:
                st.write(i)
        st.write("")

def model():        
        o = movie_title
        opt = pd.concat([pd.Series(['Select an Option']),o])
#         dropdown = widgets.Dropdown(
#             options= opt,
#             description='Chose your favourite Movie:',
#             disabled=False,
#         )
#         btn = widgets.Button(description='Find recommended movies')
#         display(dropdown)
#         display(btn)
#         def btn_eventhandler(obj):
#             if(dropdown.value!='Select an Option'):
#                 getRecom(dropdown.value)
#         btn.on_click(btn_eventhandler)
        option = st.selectbox('Select your favourite movie',opt)
        getRecom(option)


# In[80]:


model()


# In[32]:


# import pickle
# filename = 'recommender.pkl'
# pickle_out = open(filename, 'wb')
# pickle.dump(recommender, pickle_out)
# pickle_out.close()


# In[33]:


# import pickle
# filename = 'recommender.pkl'
# pickle_in = open(filename, 'rb')
# loaded_model = pickle.load(pickle_in)
# loaded_model


# In[77]:





# In[ ]:





# In[ ]:




