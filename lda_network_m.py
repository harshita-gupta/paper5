#!/usr/bin/env python

# The python code in development for visualization the connections
# between topics.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.

from gensim.models import LdaModel
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import os


womenwords = ["daughter", 'ladi', 'princess', 'wife', 'women', 'woman',
              'mother', 'girl', 'sister']

#############
# cf. http://nbviewer.ipython.org/github/sgsinclair/alta/blob/master/ipynb/TopicModelling.ipynb#Graphing-Topic-Terms

def graph_terms_to_topics(lda, outfile, num_terms=25):

    # create a new graph and size it
    G = nx.Graph()
    plt.figure(figsize=(50, 50))

    # generate the edges
    for i in range(0, lda.num_topics):
        # topicLabel = "topic " + str(i)
        terms = [term for term, val in lda.show_topic(i, num_terms)]
        if "competitor" in terms:
            continue
        if "rebukescoupl" in terms:
            continue
        if "hurrylet" in terms:
            continue
        for term in terms:
            for term2 in terms:
                if term != term2:
                    if term in womenwords or term2 in womenwords:
                        G.add_edge(term, term2, color="red")
                    else:
                        G.add_edge(term, term2)

    # cf.
    # http://networkx.lanl.gov/reference/drawing.html#module-networkx.drawing.layout
    # positions for all nodes - k=0.020, iterations=30
    pos = nx.spring_layout(G, k=0.30, iterations=80000)
    # pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)


    # we'll plot topic labels and terms labels separately to have different colours
    g = G.subgraph([topic for topic, _ in pos.items() if "topic " in str(topic)])
    nx.draw_networkx_labels(g, pos, font_color='0.75')

    g = G.subgraph([term for term, _ in pos.items() if "topic " not in str(term)])
    nx.draw_networkx_labels(g, pos, font_size=25, alpha=0.9)

    # plot edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='g',
                           alpha=0.3)

    for n in G:
        #if 1 < G.degree(n) < num_terms: nx.draw_networkx_edges(G, pos, edgelist=nx.edges(G, nbunch=n), alpha=0.2)
        if n in womenwords:
            nx.draw_networkx_edges(G, pos, edgelist=nx.edges(G, nbunch=n),
                                   edge_color='r', width=1.5)

    plt.axis('off')
    plt.savefig(outfile)
    # plt.show()



if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
outdir = "networks/dense/" + path.replace("/", "-").replace("-out", "").replace("books-genji-", "")
outfile = outdir + "-network_m-%s.png" % sys.argv[2]

model = LdaModel.load(sys.argv[1])

graph_terms_to_topics(model, outfile)
