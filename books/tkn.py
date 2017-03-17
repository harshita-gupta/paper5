from stanza.nlp.corenlp import CoreNLPClient
client = CoreNLPClient(server='http://localhost:9000',
                       default_annotators=['ssplit',
                                           'tokenize'])

marquez = 'marquez/marquez'
ulysses = 'ulysses/ulysses'
emma = 'emma/emma'

with open(emma + "token", 'w') as writeFile:
    for i in range(1, 56):
        filename = emma + "%03d" % i
        with open(filename, 'r') as myfile:
            chapter = myfile.read()

        annotated = client.annotate(chapter)
        for sentence in annotated:
            for token in sentence:
                writeFile.write(token.word)
                writeFile.write("\n")
