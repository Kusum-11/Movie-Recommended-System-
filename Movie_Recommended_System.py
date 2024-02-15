# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:04:32 2024

@author: Kusum
"""

import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer

#apply steaming
def stem(text):
    y= []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

credits = pd.read_csv("D:/NITD/python/Movie recomended system/tmdb_5000_credits.csv")
movies = pd.read_csv("D:/NITD/python/Movie recomended system/tmdb_5000_movies.csv")
movies.head()
credits.head()
movies = movies.merge(credits, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.isnull().sum())
movies.dropna(inplace=True)
print(movies.isnull().sum())
print(movies.duplicated().sum())

def convert(obj):#object is in string format
    L=[]
    for i in ast.literal_eval(obj):# string object change into list
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
print(movies['genres'])

movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):#object is in string format
    L=[]
    counter =0
    for i in ast.literal_eval(obj):# string object change into list
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast']=movies['cast'].apply(convert3)

def fetch_director(obj):#object is in string format
    L=[]
    for i in ast.literal_eval(obj): # string object change into list
        if i['job']== 'Director':
            L.append(i['name'])
            break
    return L
movies['crew']=movies['crew'].apply(fetch_director)
#convert string to list so we can concatinate(overview is in string format)
movies['overview']=movies['overview'].apply(lambda x:x.split())
# remove space between the same name
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

#merge all the column
movies['tag'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['movie_id','title','tag']]

# convert tag column to list
new_df['tag']= new_df['tag'].apply(lambda x:" ".join(x))

# change text into lower case
new_df['tag']= new_df['tag'].apply(lambda x:x.lower())

# text vectorization using Bag of words in this we combine all the words and find the frequency of each word
# remove the stop words(eg. is /am/are/of....)

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tag']).toarray()# convert into numpy array

#apply steaming
new_df['tag'].apply(stem)
#print(stem('running'))
cv.get_feature_names_out()
# find the Angular distance between movies --cosine distance

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
print(similarity[1])

def recomended(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
recomended('Batman Begins')   
