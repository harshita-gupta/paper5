import unicodedata
from stanza.nlp.corenlp import CoreNLPClient
client = CoreNLPClient(server='http://localhost:9000',
                       default_annotators=['ssplit',
                                           'tokenize'])

marquez = 'marquez/marquez'
ulysses = 'ulysses/ulysses'
emma = 'emma/emma'

with open(ulysses + "token", 'w') as writeFile:
    for i in range(1, 19):
        filename = ulysses + "%02d" % i
        with open(filename, 'r') as myfile:
            chapter = unicode(myfile.read(), encoding='utf-8')
            chapter = unicodedata.normalize('NFD', chapter)
            chapter = chapter.encode('ascii', 'ignore')

        annotated = client.annotate(chapter)
        for sentence in annotated:
            for token in sentence:
                writeFile.write(token.word)
                writeFile.write("\n")
