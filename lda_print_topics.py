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

for i in range(0, lda.num_topics):
        terms = [term for term, val in lda.show_topic(i, NUMTERMS)
                 if val > valthresh]
        termstring = str(i) + " "
        for term in terms:
            termstring += term + " "
        print termstring
