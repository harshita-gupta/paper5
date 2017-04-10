import unicodedata
import nltk
import matplotlib
from nltk.probability import FreqDist
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nltk.collocations import *
import pickle
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

fromPickle = True
stemsFromPickle = True

# dr = "books/marquez"
# encoding = "latin-1"

dr = "books/genji/seidensticker/"
root = "genji"
bookfile = root + ".txt"
encoding = "utf-8"
textpickle = root + "text.pickle"
stempickle = root + "textstems.pickle"


def getText():
    if fromPickle:
        return pickle.load(open(dr + textpickle, "rb"))
    f = open(dr + bookfile, 'rU')
    raw = f.read()
    raw = unicode(raw, encoding=encoding)
    raw = unicodedata.normalize('NFD', raw)
    raw = raw.encode('ascii', 'ignore')
    tokens = nltk.word_tokenize(raw)
    mytext = nltk.Text(tokens)
    pickle.dump(mytext, open(dr + textpickle, "wb"))
    return mytext


def getTxtOfStems(txt):
    if stemsFromPickle:
        return pickle.load(open(dr + stempickle, "rb"))
    textstems = nltk.Text([porter_stemmer.stem(x) for x in mytext.tokens])
    pickle.dump(textstems, open(dr + stempickle, "wb"))
    return textstems


def dispersionPlot(txt, words):
    txt.dispersion_plot(words)


mytext = getText()
textstems = getTxtOfStems(mytext)


fdist1 = FreqDist(textstems)
print sorted([w for w in set(textstems) if fdist1[w] > 25],
    key = lambda x: fdist1[x])
# print mytext.collocations()

# bigram_measures = nltk.collocations.BigramAssocMeasures()

# finder = BigramCollocationFinder.from_words(tokens)
# scored = finder.score_ngrams(bigram_measures.raw_freq)
# print sorted(bigram for bigram, score in scored)
# sorted(finder.above_score(bigram_measures.raw_freq, 1.0 / len(tuple(nltk.bigrams(tokens)))))

 # doctest: +NORMALIZE_WHITESPACE
