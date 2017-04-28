from gensim.models import LdaModel
from gensim import corpora
import pickle
import sys
import os
import re

NUMTERMS = 50
valthresh = 0.0040

womenwords = set(['daughter', 'ladi', 'princess', 'wife', 'women', 'woman',
                  'mother', 'girl', 'sister', 'mistress'])

menwords = set(['son', 'prince', 'father', 'emperor', 'minist',
                'lord', 'man', 'husband', 'captain', 'men',
                'lordship'])

mendependentwomen = set(['daughter', 'wife', 'mother', 'sister', 'mistress'])

singularwomen = womenwords - mendependentwomen

path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
lda = LdaModel.load(sys.argv[1])

topics = 1
mentions = 0
for i in range(0, lda.num_topics):
        terms = [term for term, val in lda.show_topic(i, NUMTERMS)
                 if val > valthresh]
        if terms:
            termstring = "Topic " + str(topics) + ": "
            if not set(terms).isdisjoint(mendependentwomen):
                termstring = "MENDEPENDENT " + termstring
                mentions += 1

            topics += 1
            for term in terms:
                termstring += term + " "
            print termstring
            print ""
            print mentions, topics
