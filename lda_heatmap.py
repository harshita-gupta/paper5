#!/usr/bin/env python

# The python code used to generate the heatmaps for each text.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.

from gensim.corpora import MmCorpus
from gensim.models import LdaModel
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import os


if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]


# load model

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


# get plot labels

topic_labels = []
for i in range(no_of_topics):
    # show_topic() returns tuples (word_prob, word)
    topic_terms = [x[0] for x in model.show_topic(i, topn=10)]
    label = ""
    label += " ".join(topic_terms[:3]) + "\n"
    label += " ".join(topic_terms[3:6]) + "\n"
    label += " ".join(topic_terms[6:]) + "\n"
    topic_labels.append(label)
# print(doc_topic)
# print(doc_topic.shape)


# cf. https://de.dariah.eu/tatom/topic_model_visualization.html

if no_of_docs > 20 or no_of_topics > 20:
    # if many items, enlarge figure
    plt.figure(figsize=(100, 100))
plt.pcolor(doc_topic, norm=None, cmap='Reds')
plt.yticks(np.arange(doc_topic.shape[0]) + 1.0, doc_labels)
plt.xticks(np.arange(doc_topic.shape[1]) + 0.5, topic_labels, rotation='90')
plt.gca().invert_yaxis()
plt.colorbar(cmap='Reds')
plt.tight_layout()

plt.savefig(path + "/" + corpusname + "_heatmap.png")  #, dpi=80)
# plt.show()
