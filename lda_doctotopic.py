#!/usr/bin/env python

# The python code used to generate the heatmaps for each text.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.

import os
import sys
import numpy as np
from gensim.corpora import MmCorpus
from gensim.models import LdaModel

path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
lda = LdaModel.load(sys.argv[1])
corpus = MmCorpus(path + "/" + corpusname + ".mm")

doc_labels = []
topicnum = 0
with open(path + "/" + corpusname + "_doclabels.txt", "r") as f:
    for line in f:
        doc_labels.append("%d: %s" % (topicnum, line))
        topicnum += 1

corpus = MmCorpus(path + "/" + corpusname + ".mm")
model = LdaModel.load(sys.argv[1])

no_of_topics = model.num_topics
no_of_docs = len(doc_labels)

# get doc-topic matrix

doc_topic = np.zeros((no_of_docs, no_of_topics))

# use document bow from corpus
for doc, i in zip(corpus, range(no_of_docs)):
    # to get topic distribution from model
    topic_dist = model.__getitem__(doc)
    # topic_dist is a list of tuples (topic_id, topic_prob)
    for topic in topic_dist:
        # save topic probability
        doc_topic[i][topic[0]] = topic[1]

for i in range(len(doc_labels)):
    print doc_labels[i], doc_topic[i]
