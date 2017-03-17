import unicodedata
from stanza.nlp.corenlp import CoreNLPClient
client = CoreNLPClient(server='http://localhost:9000',
                       default_annotators=['ssplit',
                                           'tokenize'])

marquez = 'marquez/marquez'  # encoded in latin-1
ulysses = 'ulysses/ulysses'  # encoded in utf-8
emma = 'emma/emma'           # encoded in utf-8

with open(marquez + "token", 'w') as writeFile:
    for i in range(1, 21):
        filename = marquez + "%02d" % i
        with open(filename, 'r') as myfile:
            chapter = unicode(myfile.read(), encoding='latin-1')
            chapter = unicodedata.normalize('NFD', chapter)
            chapter = chapter.encode('ascii', 'ignore')

        annotated = client.annotate(chapter)
        for sentence in annotated:
            for token in sentence:
                writeFile.write(token.word.encode('ascii', 'ignore'))
                writeFile.write("\n")
