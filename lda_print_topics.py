from gensim.models import LdaModel
import sys
import os

NUMTERMS = 50
valthresh = 0.0040

womenwords = set(['daughter', 'ladi', 'princess', 'wife', 'women', 'woman',
                  'mother', 'girl', 'sister', 'mistress'])

path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
lda = LdaModel.load(sys.argv[1])

topics = 1
for i in range(0, lda.num_topics):
        terms = [term for term, val in lda.show_topic(i, NUMTERMS)
                 if val > valthresh]
        if terms:
            termstring = "Topic " + str(topics) + ": "
            topics += 1
            for term in terms:
                termstring += term + " "
            print termstring
            print ""
