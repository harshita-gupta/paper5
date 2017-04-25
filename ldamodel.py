# The python code used to train the LDA topic model and generate
# the topics for each text.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.


from nltk.collocations import *
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
import pickle
import unicodedata
import nltk
import matplotlib
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
import sys
import os
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
enstop = set(get_stop_words('en'))
exclude = set(["genji", "kiritsubo", "suzako", "kokiden", "fujitsubo",
               "kannon", "nakanokimi", "ichijo", "yamato", "jo", "chu"
               "omyobu", "sadaijin", "chiunagon", "koremitz", "tayu",
               "yugiri", "akikonomu", "ise", "kiri", "jijiu", "saigu",
               "dainagon", "kokimi", "yokawa", "kosaisho", "ikaga",
               "chujo", "aoi", "kii", "iyo", "utsusemi", "kogimi", "nokiba",
               "koremitsu", "yugao", "ukon", "rokujo", "murasaki",
               "shonagon", "jiju", "chiujio", "shionagon", "kerria",
               "kitayama", "amagimi", "hyobu", "hitachi", "akashi",
               "tamakazura", "kashiwagi", "kaoru", "oigimi", "naka",
               "ukifune", "ono", "hachinomiya", "im", "ive", "japonica",
               "sochinomiya", "niou", "suzaku", "kumoinokari", "reizei",
               "asagao", "gosechi", "roku", "kimi", "taifu", "bennokimi",
               "ujinot", "uji", "niou", "s", "tamakazura", "kemari",
               "nakanobu", "asaka", "oborozukiyo", "kawachi", "kana", "shosho",
               "wakana", "yugei", "izumo", "tatsuta", "moku", "tsukushi",
               "kosaisho", "michisada", "sachuben", "kiyomidz", "mogi",
               "wistaria", "udaijin", "hiyojin", "tsubo", "jio",
               "azechi", "kojiju", "korabu", "kurabu", "tamakatsura",
               "koshosho", "ivi", "tokitaka", "ryozen", "kurodo", "oborozuki",
               "naishi", "hatsuse", "hachi", "sakon", "hotaru", "oyama",
               "tokikata", "yoshikiyo", "kami", "sama", "hikal", "shioshio",
               "udaiben", "kwannon", "kiyomidz", "koki", "kagura", "saisho",
               "sumiyoshi", "suruga", "yukihira", "ariwara", "kyushu",
               "umetsubo", "agemaki", "kozeri", "suyetsumu", "kobai",
               "himegimi", "higekuro", "omoto", "ochiba", "koma", "oshio",
               "seki", "shikibu", "mikawa", "sanjo", "chunagon", "izumi",
               "myobu"]).union(enstop)

fromPickle = int(sys.argv[8]) == 1
stemsFromPickle = int(sys.argv[9]) == 1
modifiedReduce = int(sys.argv[10]) == 1
noreduce = int(sys.argv[10]) == 99
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
#  takes tagged and stemmed words and returns tokens that match requirements.
def getReducedTokens(tags, dr, root):
    if noreduce:
        tkns = [x[2] for x in tags if x[0] not in enstop and
                x[2] not in enstop]
    elif modifiedReduce:
        tkns = [x[2] for x in tags if (x[1] == "NN" or x[1] == "NNS" or
                x[1] == "RB" or x[1] == 'RBR' or x[1] == 'RBS' or
                x[1] == 'VB' or x[1] == 'VBG' or x[1] == 'VBD' or
                x[1] == 'VBN' or
                x[1] == 'VBP' or x[1] == 'VBZ' or x[1] == 'JJ' or
                x[1] == 'JJR' or x[1] == 'JJS') and
                x[0] not in exclude and x[2] not in exclude]
    else:
        tkns = [x[2] for x in tags if
                x[1] != 'PRP' and x[1] != 'PRP$' and
                x[1] != 'NNP' and x[1] != 'NNPS' and
                x[1] != "FW" and
                x[0] not in exclude and x[2] not in exclude]
    stems = nltk.Text(tkns)
    return stems


def dispersionPlot(txt, words):
    txt.dispersion_plot(words)


def outputTokenFile():
    with open(dr + "token.txt", 'w') as tokenfile:
        for item in mytext.tokens:
            tokenfile.write("%s " % item)


def splitListIntoWindows(lst, wlen):
    # want to return multiple documents with each window length
    if overlap == 0:
        windows = [lst[i:i + wlen] for i in range(0, len(lst), wlen)]
    else:
        windows = [sb for sb in
                   (lst[x:x + wlen] for x in range(len(lst) - wlen + 1))]
    return windows


reducedpickle = root + "reducedtext%d.pickle" % overlap
if stemsFromPickle:
    logging.info("loading reduced and cleaned slices from pickle...")
    reducedSlices = pickle.load(open(dr + reducedpickle, "rb"))
    stemmedDict = pickle.load(open(dr + root + "stemmeddict.pickle", "rb"))
    doc_labels = pickle.load(open(dr + root + "doc_labels.pickle", "rb"))
    logging.info("slices loaded.")
else:
    logging.info("generating text...")
    regularText = getText(dr, root, encoding)
    logging.info("text generated.")

    logging.info("tagging text...")
    taggedText = nltk.pos_tag(regularText.tokens)
    logging.info("text tagged.")

    logging.info("stemming text...")
    stemmedText = []
    stemmedDict = []
    for token in taggedText:
        tk, pos = token
        stm = stemmer.stem(tk)
        stemmedText.append((tk, pos, stm))
        stemmedDict.append(stm)
    logging.info("text stemmed.")
    pickle.dump(stemmedDict, open(dr + root + "stemmeddict.pickle", "wb"))

    logging.info("splitting text into windows...")
    slices = splitListIntoWindows(stemmedText, WINDOWS)
    logging.info("windows generated.")
    logging.info("reducing and cleaning slices...")

    doc_labels = []
    for window in slices:
        label = ""
        for word in window[:5]:
            label += word[0] + " "
        doc_labels.append(label)
    pickle.dump(doc_labels, open(dr + "doc_labels.pickle", "wb"))

    reducedSlices = [getReducedTokens(sl, dr, root) for sl in slices]
    pickle.dump(reducedSlices, open(dr + reducedpickle, "wb"))
    logging.info("slices reduced and cleaned.")

logging.info("creating dictionary...")
dictionary = corpora.Dictionary(reducedSlices)
logging.info("dictionary created.")

logging.info("creating corpus...")
corpus = [dictionary.doc2bow(sl.tokens) for sl in reducedSlices]
logging.info("corpus created.")

logging.info("generating lda model...")
ldamodel = models.ldamulticore.LdaMulticore(corpus, num_topics=TOPICS,
                                            id2word=dictionary, passes=PASSES,
                                            workers=3)
logging.info("LDA model generated.")
pickle.dump(ldamodel,
            open(dr + "ldamodel_overlap%d_w%d_t%d_p%d.pickle" %
                 (overlap, WINDOWS, TOPICS, PASSES), "wb"))

topics = ldamodel.show_topics(num_topics=TOPICS)

print("saving ...\n")

if not os.path.exists(dr + "out"):
    os.makedirs(dr + "out")

foldername = dr + "out/ldamodel_overlap%d_w%d_t%d_p%d_modified%d" % (overlap,
                                                        WINDOWS,
                                                          TOPICS, PASSES,
                                                          modifiedReduce)

with open(foldername + "_doclabels.txt", "w") as f:
    for item in doc_labels:
        f.write(item + "\n")

with open(foldername + "_topics.txt", "w") as f:
    for item, i in zip(topics, enumerate(topics)):
        f.write("topic #" + str(i[0]) + ": " + str(item) + "\n")

dictionary.save(foldername + ".dict")
corpora.MmCorpus.serialize(foldername + ".mm", corpus)
ldamodel.save(foldername + ".lda")
