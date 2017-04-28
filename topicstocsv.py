#!/usr/bin/env python

# The python code to turn the lda model's computed topics into a CSV file for
# cloud generation.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.


from gensim.models import LdaModel
import sys
import os

if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
outfile = path + "/" + corpusname + "wordclouds/"

model = LdaModel.load(sys.argv[1])

if not os.path.exists(outfile):
    os.makedirs(outfile)

for i in range(0, model.num_topics):
    with open(outfile + "topic_words%02d.csv" % i, "wb") as f:
        f.write("text,size,topic\n")
        terms = [(val, term) for val, term in model.show_topic(i, 25)]
        for (val, term) in terms:
            f.write("%s,%d,%d\n" % (val, int(term * 100000), i))
