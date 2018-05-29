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
from sklearn.metrics import silhouette_samples, silhouette_score

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
    
    
num_clusters = 15
max_iterations = 3000

# calculate tf-idf of texts
s = set(stopwords.words('english'))

#my_stop_words = text.ENGLISH_STOP_WORDS.union(s).union(["bioinformatics","science","berlin","verlag","heidelberg","ms","copyright","ieee","society","biomed","springer","elsevier","factor","change","data","central","relationship","term","nucleic","acid","research","right","conclusions","validation","site","step","size","selection","vector","behalf","target","work","context","type","domain","power","licensee","license","identification","background","conclusion","role","difference","problem","model","occurrence","compraison","similarity","level","search","state","body","connectivity","distribution","random","graph","requirement","control","normalization","normal","chosen","output","regression","class","time","space","rate","performance","procedure","process","interface","scale","matrix","characteristic","accession","need","determination","error","date","addition","completeness","risk","probe","help","line","image","segmentation","measure","transformation","execution","result","results","query","web","investigation","investigate","evidence","profile","interaction","knowledge","activity","browser","scheme","corpus","ontology","support","lack","project","user","feedback","technology","template","assignment","server","platform","manner","management","storage","cross","classification","classifier","prediction","predictor","funcionality","failure","study","ii","iii","i","sum","access","progress","information","run","batch","literature","www","html","xml","research","language","log","feature","ratio","quality","spot","order","test","http","org","com","range","text","significance","oxford","set","accuracy","use","paper","method","published","rights","press","university","new","based","number","large","developed","reserved","motivation","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","author","authors","used","using","approach","summary","results"])
my_stop_words = text.ENGLISH_STOP_WORDS.union(s).union(["event","technique","application","graphic","graphics","day","method","methods","acm","format","mean","source","scientist","issue","subject","area","java","researcher","year","challenge","resource","case","advance","solution","taylor","francis","kegg","patient","gpu","parameter","value","bioinformatics","science","berlin","verlag","heidelberg","ms","copyright","ieee","society","biomed","springer","elsevier","factor","change","data","central","relationship","term","nucleic","acid","research","right","conclusions","validation","site","step","size","selection","vector","behalf","target","work","context","type","domain","power","licensee","license","identification","background","conclusion","role","difference","problem","model","occurrence","compraison","similarity","level","search","state","body","connectivity","distribution","random","graph","requirement","control","normalization","normal","chosen","output","regression","class","time","space","rate","performance","procedure","process","interface","scale","matrix","characteristic","accession","need","determination","error","date","addition","completeness","risk","probe","help","line","image","segmentation","measure","transformation","execution","result","results","query","web","investigation","investigate","evidence","profile","interaction","knowledge","activity","browser","scheme","corpus","ontology","support","lack","project","user","feedback","technology","template","assignment","server","platform","manner","management","storage","cross","classification","classifier","prediction","predictor","funcionality","failure","study","ii","iii","i","sum","access","progress","information","run","batch","literature","www","html","xml","research","language","log","feature","ratio","quality","spot","order","test","http","org","com","range","text","significance","oxford","set","accuracy","use","paper","method","published","rights","press","university","new","based","number","large","developed","reserved","motivation","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","author","authors","used","using","approach","summary","results"])

# Open database connection
db = MySQLdb.connect("","","","" )

cursor = db.cursor()

# Fetch a single row using fetchone() method.
wordnet_lemmatizer = WordNetLemmatizer()

# Load them from file
title_list = cPickle.load(open("title_list2.pickle", "rb"))
print(len(title_list))
tf_idf_vectorizer = cPickle.load(open("tfidfvectorizer2.pickle", "rb"))
tf_idf_matrix = cPickle.load(open("tfidfmatrix2.pickle", "rb"))
X_dense = tf_idf_matrix.todense()
texts_list = cPickle.load(open("texts_list2.pickle", "rb"))
print(len(texts_list))
#
print("TDF Vectorizer, matrix and texts loaded from file")

range_n_clusters = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_dense) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iterations, init='k-means++', max_no_improvement=10, n_init=10, batch_size=45, verbose=False)
    cluster_labels = clusterer.fit_predict(X_dense)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_dense, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_dense, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter([X_dense[:, 0]], [X_dense[:, 1]], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

clustering_model = MiniBatchKMeans(n_clusters=num_clusters, max_iter=max_iterations, init='k-means++', max_no_improvement=10, n_init=10, batch_size=45, verbose=False)
clustering_model.fit(X_dense)
