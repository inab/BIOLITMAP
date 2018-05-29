#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abazaga
"""

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt, numpy as np

def untick(sub):
    sub.tick_params(which='both', bottom='off', top='off',  labelbottom='off', labelleft='off', left='off', right='off')

digits = load_digits()



images = digits['images']
images = [image.reshape((1,-1)) for image in images]
images = np.concatenate(tuple(images), axis = 0)

topicsRange = [(i+1)*5 for i in range(10)]
print(topicsRange)

ldaModels = [LDA(n_components = numTopics, learning_method='batch') for numTopics in topicsRange]

for lda in ldaModels:
    lda.fit(images)

scores = [lda.perplexity(images) for lda in ldaModels]

plt.plot(topicsRange, scores)
plt.show()

maxLogLikelihoodTopicsNumber = np.argmin(scores)
plotNumbers = [4, 14, 24, 34, 44, 49]

if maxLogLikelihoodTopicsNumber not in plotNumbers:
    plotNumbers.append(maxLogLikelihoodTopicsNumber)

for numberOfTopics in plotNumbers:
    plt.figure()
    modelIdx = topicsRange.index(numberOfTopics)
    lda = ldaModels[modelIdx]
    sideLen = int(np.ceil(np.sqrt(numberOfTopics)))
    for topicIdx, topic in enumerate(lda.components_):
        ax = plt.subplot(sideLen, sideLen, topicIdx + 1)
        ax.imshow(topic.reshape((8,8)), cmap = plt.cm.gray_r)
        untick(ax)
    plt.show()