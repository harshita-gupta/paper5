from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import sys
import unicodedata
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
from numpy import var as variance, mean, median
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

sid = SentimentIntensityAnalyzer()
fromPickle = True


def getText(dr, root='genji', encoding='utf-8'):
    bookfile = root + ".txt"
    f = open(dr + bookfile, 'rU')
    raw = f.read()
    raw = unicode(raw, encoding=encoding)
    raw = unicodedata.normalize('NFD', raw)
    raw = raw.encode('ascii', 'ignore')
    return raw.lower()


text = getText('books/genji/' + sys.argv[1] + '/')

# lines_list = sent_tokenize(text)
# sentiments = []
# for sentence in lines_list:
#     ss = sid.polarity_scores(sentence)
#     if abs(ss['compound']) > 0.00001:
#         sentiments.append((sentence, ss['compound']))

# compounds = numpy.array([pair[1] for pair in sentiments])

# print sum(compounds) / len(sentiments)
# print variance(compounds)
# print mean(compounds)
# print median(compounds)
    # for k in sorted(ss):
        # print '{0}: {1}, '.format(k, ss[k])

words = tokenizer.tokenize(text)
wordsents = []
for word in words:
    ss = sid.polarity_scores(word)
    if abs(ss['compound']) > 0.00001:
        wordsents.append((stemmer.stem(word), ss['compound']))

std = sorted(set(wordsents), key=lambda x: x[1])
for w in std[:20]:
    print w

# plt.hist(compounds, bins=len(sentiments) / 100)
# axes = plt.gca()
# axes.set_ylim([0,1000])
# plt.title("Sentiment Distribution Across %s's Genji" % sys.argv[1].title())
# plt.show()
