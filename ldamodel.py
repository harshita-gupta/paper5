from nltk.collocations import *
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
import pickle
import unicodedata
import nltk
import matplotlib
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from optparse import OptionParser
from nltk.probability import FreqDist
import sys
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# parser = OptionParser()
# parser.add_option("-dr", "--directory", dest="dr",
#                   help="", metavar="FILE")
# parser.add_option("-r", "--root", dest="root",
#                   help="", metavar="FILE")
# parser.add_option("-q", "--quiet",
#                   action="store_false", dest="verbose", default=True,
#                   help="don't print status messages to stdout")

# (options, args) = parser.parse_args()


dr = sys.argv[1]  ## "books/genji/washburn/"
root = sys.argv[2]  ## "genji"
encoding = sys.argv[3]  ## "utf-8"

WINDOWS = int(sys.argv[4])
TOPICS = int(sys.argv[5])
PASSES = int(sys.argv[6])
overlap = int(sys.argv[7])

tokenizer = RegexpTokenizer(r'\w+')

stemmer = PorterStemmer()
enstop = get_stop_words('en')
genjinames = set(["genji", "kiritsubo", "suzako", "kokiden", "fujitsubo",
                  "omyobu",
                  "chujo", "aoi", "kii", "iyo", "utsusemi", "kogimi", "nokiba",
                  "koremitsu", "yugao", "ukon", "rokujo", "murasaki",
                  "shonagon", "jiju",
                  "kitayama", "amagimi", "hyobu", "hitachi", "akashi",
                  "tamakazura", "kashiwagi", "kaoru", "oigimi", "naka",
                  "ukifune", "ono", "hachinomiya", "im", "ive",
                  "sochinomiya", "niou", "suzaku", "kumoinokari", "reizei",
                  "asagao", "gosechi", "roku", "kimi", "taifu"])

fromPickle = True
stemsFromPickle = True

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
    tokens = tokenizer.tokenize(raw.lower())
    mytext = nltk.Text(tokens)
    pickle.dump(mytext, open(dr + textpickle, "wb"))
    return mytext


#  Removed stop words, proper nouns, and stems all results.
def getReducedText(tokens, dr, root):
    reducedpickle = root + "reducedtext.pickle"
    if stemsFromPickle:
        return pickle.load(open(dr + reducedpickle, "rb"))
    tags = nltk.pos_tag(tokens)
    tkns = [x[0] for x in tags if
            x[0] not in enstop and x[0] not in genjinames and
            x[1] != 'PRP' and x[1] != 'PRP$' and
            x[1] != 'NNP' and x[1] != 'NNPS']
    txttokens = [stemmer.stem(x) for x in tkns if x not in enstop and
                 x not in genjinames]
    stems = nltk.Text(txttokens)
    pickle.dump(stems, open(dr + reducedpickle, "wb"))
    return stems


def dispersionPlot(txt, words):
    txt.dispersion_plot(words)


def outputTokenFile():
    with open(dr + "token.txt", 'w') as tokenfile:
        for item in mytext.tokens:
            tokenfile.write("%s " % item)


def splitTxtIntoWindows(txt, wlen):
    # want to return multiple documents with each window length
    tkns = txt.tokens
    if overlap == 0:
        windows = [tkns[i:i + wlen] for i in range(0, len(tkns), wlen)]
    else:
        windows = [sb for sb in
                   (tkns[x:x + wlen] for x in range(len(tkns) - wlen + 1))]
    return windows


print "generating text..."
regularText = getText(dr, root, encoding)
print "text generated"

print "splitting text into windows..."
slices = splitTxtIntoWindows(regularText, WINDOWS)
print "windows generated."

print "reducing and cleaning slices..."
reducedSlices = [getReducedText(sl, dr, root) for sl in slices]
print "slices reduced and cleaned."

print "creating dictionary..."
dictionary = corpora.Dictionary([getReducedText(txt, dr, root)])
print "dictionary created."

print "creating corpus..."
corpus = [dictionary.doc2bow(reducedSlice) for reducedSlice in reducedSlices]
print "corpus created."

print "generating lda model..."
ldamodel = models.ldamulticore.LdaMulticore(corpus, num_topics=TOPICS,
                                            id2word=dictionary, passes=PASSES,
                                            workers=3)
print "LDA model generated."
pickle.dump(ldamodel,
            open(dr + "ldamodel_overlap%d_w%d_t%d_p%d.pickle" %
                 (overlap, WINDOWS, TOPICS, PASSES), "wb"))
print(ldamodel.print_topics(num_topics=TOPICS, num_words=20))

print("saving ...\n")

if not os.path.exists("out"):
    os.makedirs("out")

foldername = dr

with open("out/" + foldername + "_doclabels.txt", "w") as f:
    for item in doc_labels:
        f.write(item + "\n")

with open("out/" + dr + "_topics.txt", "w") as f:
    for item, i in zip(topics, enumerate(topics)):
        f.write("topic #" + str(i[0]) + ": " + str(item) + "\n")

dictionary.save("out/" + foldername + ".dict")
MmCorpus.serialize("out/" + foldername + ".mm", corpus)
model.save("out/" + foldername + ".lda")


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
