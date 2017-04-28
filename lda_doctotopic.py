#!/usr/bin/env python

# The python code used to generate the heatmaps for each text.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.

from nltk.stem.porter import PorterStemmer
import os
import unicodedata
import sys
from gensim import corpora
from gensim.corpora import MmCorpus
from gensim.models import LdaModel
from nltk.tokenize import RegexpTokenizer
import nltk
from stop_words import get_stop_words

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

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
lda = LdaModel.load(sys.argv[1])
homefolder, _ = os.path.split(path)
_, translator = os.path.split(homefolder)
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


def getReducedTokens(tags):
    tkns = [x[2] for x in tags if (x[1] == "NN" or x[1] == "NNS" or
            x[1] == "RB" or x[1] == 'RBR' or x[1] == 'RBS' or
            x[1] == 'VB' or x[1] == 'VBG' or x[1] == 'VBD' or
            x[1] == 'VBN' or
            x[1] == 'VBP' or x[1] == 'VBZ' or x[1] == 'JJ' or
            x[1] == 'JJR' or x[1] == 'JJS') and
            x[0] not in exclude and x[2] not in exclude]
    stems = nltk.Text(tkns)
    return stems


f = open(translator + "-consummation.txt", 'rU')
raw = f.read()
raw = unicode(raw, encoding='utf-8')
raw = unicodedata.normalize('NFD', raw)
raw = raw.encode('ascii', 'ignore')
tokens = tokenizer.tokenize(raw.lower())
mytext = nltk.Text(tokens)
taggedText = nltk.pos_tag(mytext.tokens)
stemmedText = []
for token in taggedText:
        tk, pos = token
        stm = stemmer.stem(tk)
        stemmedText.append((tk, pos, stm))

reduced = getReducedTokens(stemmedText)
dictionary = corpora.Dictionary([reduced])

# tops = lda.top_topics(corpus, 15)

# for topic in tops:
#     print topic

print [(t + 1, p) for (t, p) in lda.get_document_topics(dictionary.doc2bow(reduced), 0.001)]

# topicpredictions = lda.get_term_topics(sys.argv[2], 0.0001)
# p = sorted([(t, v * 100) for t, v in topicpredictions], key=lambda x: x[1],
#             reverse=True)
# for t, v in p:
#     print "%d, %02f" % (t, v)
