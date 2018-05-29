#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:25:54 2017

@author: abazaga
"""

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import glob
from ggplot import *
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import MySQLdb
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from nltk.cluster.kmeans import *
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk import *
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import text
from itertools import cycle, islice
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.random_projection import sparse_random_matrix
import nltk.stem
import re
from matplotlib import cm
import matplotlib.colors

import _pickle as cPickle

pattern = re.compile("^([A-Z][0-9]+)+$")

stemmer = EnglishStemmer()
analyzernew = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzernew(doc))

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" | ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
# Open database connection
db = MySQLdb.connect("","","","" )

cursor = db.cursor()

cursor.execute("SELECT abstract,title from - WHERE type IN ('Article','Conference Paper') ORDER BY RAND()")

wordnet_lemmatizer = WordNetLemmatizer()
texts_list = []
title_list = []
for row in cursor:
    textToFilter = str(row[0])
    title_list.append(str(row[1]))
    finalText = ""
    tok=nltk.tokenize.word_tokenize(textToFilter)
    pos=nltk.pos_tag(tok)
    for word in pos:
        if word[1].startswith("N") and len(word[1]) > 2 and not word[1].endswith("ing"):
            tempWord = word[0]
            tempWord =wordnet_lemmatizer.lemmatize(tempWord)
            if pattern.match(tempWord):
                finalText = finalText
            else:
                finalText+=tempWord + " "
                
    finalText = re.sub(r"\bsnp\b","snps",finalText)
    finalText = re.sub(r"\bgene\b","genes",finalText)
    finalText = re.sub(r"\brnas\b","rna",finalText)
    finalText = re.sub(r"\bseq\b","sequence",finalText)
    finalText = re.sub(r"\bmarker\b","biomarkers",finalText)
    finalText = re.sub(r"\bmarkers\b","biomarkers",finalText)
    finalText = re.sub(r"\bbiomarker\b","biomarkers",finalText)
    texts_list.append(finalText[:-1])

num_clusters = 20
num_seeds = 10
max_iterations = 3000

labels_color_map = {
    0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'orange',
    5: 'pink', 6: 'purple', 7: 'cyan', 8: 'sienna', 9: 'black',
    10: 'lightseagreen', 11: 'fuchsia'
}

pca_num_components = 2
tsne_num_components = 3


s = set(stopwords.words('english'))
#my_stop_words = text.ENGLISH_STOP_WORDS.union(s).union(["bioinformatics","science","berlin","verlag","heidelberg","ms","copyright","ieee","society","biomed","springer","elsevier","factor","change","data","central","relationship","term","nucleic","acid","research","right","conclusions","validation","site","step","size","selection","vector","behalf","target","work","context","type","domain","power","licensee","license","identification","background","conclusion","role","difference","problem","model","occurrence","compraison","similarity","level","search","state","body","connectivity","distribution","random","graph","requirement","control","normalization","normal","chosen","output","regression","class","time","space","rate","performance","procedure","process","interface","scale","matrix","characteristic","accession","need","determination","error","date","addition","completeness","risk","probe","help","line","image","segmentation","measure","transformation","execution","result","results","query","web","investigation","investigate","evidence","profile","interaction","knowledge","activity","browser","scheme","corpus","ontology","support","lack","project","user","feedback","technology","template","assignment","server","platform","manner","management","storage","cross","classification","classifier","prediction","predictor","funcionality","failure","study","ii","iii","i","sum","access","progress","information","run","batch","literature","www","html","xml","research","language","log","feature","ratio","quality","spot","order","test","http","org","com","range","text","significance","oxford","set","accuracy","use","paper","method","published","rights","press","university","new","based","number","large","developed","reserved","motivation","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","author","authors","used","using","approach","summary","results"])
my_stop_words = text.ENGLISH_STOP_WORDS.union(s).union(["event","technique","application","graphic","graphics","day","method","methods","acm","format","mean","source","scientist","issue","subject","area","java","researcher","year","challenge","resource","case","advance","solution","taylor","francis","kegg","patient","gpu","parameter","value","bioinformatics","science","berlin","verlag","heidelberg","ms","copyright","ieee","society","biomed","springer","elsevier","factor","change","data","central","relationship","term","nucleic","acid","research","right","conclusions","validation","site","step","size","selection","vector","behalf","target","work","context","type","domain","power","licensee","license","identification","background","conclusion","role","difference","problem","model","occurrence","compraison","similarity","level","search","state","body","connectivity","distribution","random","graph","requirement","control","normalization","normal","chosen","output","regression","class","time","space","rate","performance","procedure","process","interface","scale","matrix","characteristic","accession","need","determination","error","date","addition","completeness","risk","probe","help","line","image","segmentation","measure","transformation","execution","result","results","query","web","investigation","investigate","evidence","profile","interaction","knowledge","activity","browser","scheme","corpus","ontology","support","lack","project","user","feedback","technology","template","assignment","server","platform","manner","management","storage","cross","classification","classifier","prediction","predictor","funcionality","failure","study","ii","iii","i","sum","access","progress","information","run","batch","literature","www","html","xml","research","language","log","feature","ratio","quality","spot","order","test","http","org","com","range","text","significance","oxford","set","accuracy","use","paper","method","published","rights","press","university","new","based","number","large","developed","reserved","motivation","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","author","authors","used","using","approach","summary","results"])

tf_idf_vectorizer = TfidfVectorizer(stop_words=my_stop_words, ngram_range=(1, 2), analyzer="word",max_df=0.3, min_df=0.05)
tf_idf_matrix = tf_idf_vectorizer.fit_transform(texts_list)

# Save them to file
cPickle.dump(tf_idf_vectorizer, open("tfidfvectorizer.pickle", "wb"))
cPickle.dump(tf_idf_matrix, open("tfidfmatrix.pickle", "wb"))
cPickle.dump(texts_list, open("texts_list.pickle", "wb"))
#
print("TDF Vectorizer, matrix and texts saved to file")

similarity_distance = 1 - cosine_similarity(tf_idf_matrix)

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.3, min_df=0.05, stop_words=my_stop_words)
tf = tf_vectorizer.fit_transform(texts_list)
tf_feature_names = tf_vectorizer.get_feature_names()

clustering_model = MiniBatchKMeans(n_clusters=num_clusters, max_iter=max_iterations, init='k-means++', max_no_improvement=10, n_init=10, batch_size=45, verbose=False)

clustering_model.fit(tf_idf_matrix)

clusters = {}
n = 0
for item in clustering_model.labels_:
    if item in clusters:
        clusters[item].append(texts_list[n])
    else:
        clusters[item] = [texts_list[n]]
    
    cursor = db.cursor()
    cursor.execute ("UPDATE articlesmulti3 SET topic = %s WHERE title=%s", (item,title_list[n]))
    db.commit()
    
    n +=1
    
    

db.close()

y_kmeans = clustering_model.predict(tf_idf_matrix)
for item in clusters:
    print("Cluster ", item)
    print("Size: ", len(clusters[item]))

centers = clustering_model.cluster_centers_

svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
svd.fit(tf_idf_matrix)
centers2D = svd.inverse_transform(centers)

plt.hold(True)
plt.scatter(centers2D[:,0], centers2D[:,1], 
            marker='x', s=200, linewidths=3, c='r')
plt.show()