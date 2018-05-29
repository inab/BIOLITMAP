#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:08:25 2017

@author: abazaga
"""

import MySQLdb
import spacy
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

nlp = spacy.load('/home/abazaga/.local/lib/python3.5/site-packages/spacy/data/en')

# Open database connection
db = MySQLdb.connect("","","","" )

cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT name,id from places")

texts_list_orig = []
for row in cursor:
    texts_list_orig.append(row)
    
cursor.execute("SELECT name from places")
import re

def f(s, pat):
    pat = r'(\w*%s\w*)' % pat       
    return re.findall(pat, s)
    
texts_list = []
for row in cursor:
    list_found_to_remove = f(row[0],"Univer")
    list_found_to_remove.append(f(row[0],"Insti"))
    list_found_to_remove.append(f(row[0],"Cent"))
    list_found_to_remove.append(f(row[0],"Lab"))
    list_found_to_remove.append(f(row[0],"Scho"))
    list_found_to_remove.append(f(row[0],"Colle"))
    list_found_to_remove.append(f(row[0],"Stud"))
    list_found_to_remove.append(f(row[0],"Acade"))
    list_found_to_remove.append(f(row[0],"Akade"))
    list_found_to_remove.append(f(row[0],"Hospi"))
    list_found_to_remove.append(f(row[0],"NHS"))
    list_found_to_remove.append(f(row[0],"Muse"))
    list_found_to_remove.append(f(row[0],"Agen"))
    list_found_to_remove.append(f(row[0],"Ecole"))
    list_found_to_remove.append(f(row[0],"Health"))
    list_found_to_remove.append(f(row[0],"Divis"))
    newRow = row[0]
    for array in list_found_to_remove:
        for word in array:
            newRow.replace(word,'')
    texts_list.append(newRow)


cachedStopWordsEng = stopwords.words("english")
cachedStopWordsFr = stopwords.words("french")
cachedStopWordsDe = stopwords.words("german")
cachedStopWordsIt = stopwords.words("italian")
cachedStopWordsRu = stopwords.words("russian")
totalCachedStopWords = cachedStopWordsEng + cachedStopWordsFr + cachedStopWordsDe + cachedStopWordsIt + cachedStopWordsRu

vectorizer = CountVectorizer(stop_words = totalCachedStopWords)


transformer = TfidfTransformer()
trainVectorizerArray = vectorizer.fit_transform(texts_list).toarray()
testVectorizerArray = vectorizer.transform(texts_list).toarray()

cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

duplicatedNames = []

for idObservado, vector in enumerate(trainVectorizerArray):
    if(texts_list_orig[idObservado][0] in duplicatedNames):
        continue
    
    duplicatedNames.append(texts_list_orig[idObservado][0])
   
    for idSegundo, vector2 in enumerate(testVectorizerArray):
        if(idObservado == idSegundo):
            continue
        
        cosine = cx(vector,vector2)
        if cosine > 0.7:
            print(cosine)
            print(texts_list_orig[idSegundo][0])
            cursor = db.cursor()
            cursor.execute ("UPDATE places SET realId = %s WHERE id=%s", (texts_list_orig[idObservado][1],texts_list_orig[idSegundo][1]))
            db.commit()
            duplicatedNames.append(texts_list_orig[idSegundo][0])