from nltk.collocations import *
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
import pickle
import unicodedata
import nltk
import matplotlib
from gensim import corpora, models
from nltk.probability import FreqDist
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

stemmer = PorterStemmer()
enstop = get_stop_words('en')

fromPickle = True
stemsFromPickle = True

# dr = "books/marquez"
# encoding = "latin-1"

dr = "books/genji/washburn/"
root = "genji"
encoding = "utf-8"

translations = []


def getText(dr, root, encoding):
    bookfile = root + ".txt"
    textpickle = root + "text.pickle"
    if fromPickle:
        return pickle.load(open(dr + textpickle, "rb"))
    f = open(dr + bookfile, 'rU')
    raw = f.read()
    raw = unicode(raw, encoding=encoding)
    raw = unicodedata.normalize('NFD', raw)
    raw = raw.encode('ascii', 'ignore')
    tokens = tokenizer.tokenize(raw)
    mytext = nltk.Text(tokens)
    pickle.dump(mytext, open(dr + textpickle, "wb"))
    return mytext


#  also removes stop words
def getTxtOfStems(txt, dr, root):
    stempickle = root + "textstems.pickle"
    if stemsFromPickle:
        return pickle.load(open(dr + stempickle, "rb"))
    stems = nltk.Text([stemmer.stem(x) for x in txt.tokens if x not in enstop])
    pickle.dump(stems, open(dr + stempickle, "wb"))
    return stems


def cleanText(dr, root, encoding):
    return getTxtOfStems(getText(dr, root, encoding), dr, root)


def dispersionPlot(txt, words):
    txt.dispersion_plot(words)


def outputTokenFile():
    with open(dr + "token.txt", 'w') as tokenfile:
        for item in mytext.tokens:
            tokenfile.write("%s " % item)


def splitTxtIntoWindows(txt, wlen):
    # want to return multiple documents with each window length
    tags = nltk.pos_tag(txt.tokens)
    tkns = [x[0] for x in tags if x[1] != 'PRP' and x[1] != 'PRP$' and
            x[1] != 'NNP']
    # windows = [tkns[i:i + wlen] for i in range(0, len(tkns), wlen)]
    windows = [sb for sb in
               (tkns[x:x + wlen] for x in range(len(tkns) - wlen + 1))]
    return windows


dr = "books/genji/washburn/"
translations = splitTxtIntoWindows(
    cleanText(dr, "genji", "utf-8"), 1000)
# translations.append(cleanText("books/genji/washburn/", "genji", "utf-8"))


# translations.append(cleanText("books/genji/seidensticker/", "genji", "utf-8"))
# translations.append(cleanText("books/genji/kencho/", "genji", "utf-8"))

dictionary = corpora.Dictionary(translations)
corpus = [dictionary.doc2bow(translation) for translation in translations]
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=20,
                                    id2word=dictionary, passes=10)
pickle.dump(ldamodel, open(dr + "ldamodel_overlap_1000w_20t_10p.pickle", "wb"))
print(ldamodel.print_topics(num_topics=20, num_words=20))


# fdist1 = FreqDist(textstems)

# print sorted([w for w in set(textstems) if fdist1[w] > 25],
#     key = lambda x: fdist1[x])
# print mytext.collocations()

# bigram_measures = nltk.collocations.BigramAssocMeasures()

# finder = BigramCollocationFinder.from_words(tokens)
# scored = finder.score_ngrams(bigram_measures.raw_freq)
# print sorted(bigram for bigram, score in scored)
# sorted(finder.above_score(bigram_measures.raw_freq, 1.0 / len(tuple(nltk.bigrams(tokens)))))

 # doctest: +NORMALIZE_WHITESPACE
