from stanza.nlp.corenlp import CoreNLPClient
client = CoreNLPClient(server='http://localhost:9000',
                       default_annotators=['ssplit',
                                           'tokenize'])

marquez = 'marquez/marquez'
ulysses = 'ulysses/ulysses'
emma = 'emma/emma'

for i in range(1, 19):
    filename = emma + "%03d" % i
    with open(filename, 'r') as myfile:
        chapter = myfile.read()

    # book = "hello it's me. um hi."
    annotated = client.annotate(chapter)
    for sentence in annotated:
        for token in sentence:
            print token
# print annotated
# for sentence in annotated.sentences:
#     print('sentence', sentence)
#     for token in sentence:
#         print(token.word, token.lemma, token.pos, token.ner)

# from nltk.tokenize.stanford import StanfordTokenizer
# # from nltk import word_tokenize
# # from nltk.tag import StanfordPOSTagger
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

# mypath = '/Users/Harshita/Documents/Harvard/17 Freshman Spring/HUM10/paper5/'
# # mypath = mypath + 'stanford-dirs/stanford-postagger-2016-10-31/'

# # jar = mypath + 'stanford-postagger.jar'
# # model = mypath + 'models/english-left3words-distsim.tagger'

# # pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

# print StanfordTokenizer().tokenize(unicode(book))
