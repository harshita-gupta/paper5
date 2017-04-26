#!/usr/bin/env python

# The python code in development for visualization the connections
# between topics.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.

from gensim.models import LdaModel
import networkx as nx
import matplotlib
import pickle
import sys
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fromPickle = True

NUMTERMS = 50
valthresh = 0.0040

womenwords = set(['daughter', 'ladi', 'princess', 'wife', 'women', 'woman',
                  'mother', 'girl', 'sister'])

mendependentwomen = set(['daughter', 'wife', 'mother', 'sister'])

menwords = set(['son', 'prince'])

testwords = womenwords


#############
def graph_terms_to_topics(lda, outfile, num_terms=NUMTERMS):

    G = nx.Graph()
    plt.figure(figsize=(50, 50))

    for i in range(0, lda.num_topics):
        terms = [term for term, val in lda.show_topic(i, num_terms)
                 if val > valthresh]
        if "competitor" in terms:
            continue
        if "rebukescoupl" in terms:
            continue
        if "hurrylet" in terms:
            continue
        for term in terms:
            for term2 in terms:
                if term != term2:
                    if term in testwords or term2 in testwords:
                        G.add_edge(term, term2, color="red")
                    else:
                        G.add_edge(term, term2)

    if fromPickle:
        with open((outfile % "positions-") + ".d", 'wb') as f:
            pos = pickle.load(f)
    else:
        pos = nx.spring_layout(G, k=0.30, iterations=80000)
        with open((outfile % "positions-") + ".d", 'wb') as f:
            pickle.dump(pos, f, pickle.HIGHEST_PROTOCOL)

    g = G.subgraph([term for term, _ in pos.items()])
    nx.draw_networkx_labels(g, pos, font_size=25, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='g',
                           alpha=0.3)

    for n in G:
        if n in testwords:
            nx.draw_networkx_edges(G, pos, edgelist=nx.edges(G, nbunch=n),
                                   edge_color='r', weight=1.2)

    plt.axis('off')
    plt.savefig(outfile % "-with-womenwords")

    plt.clf()

    # GRAPH 2

    nodesToDel = []
    for n in G:
        if n in testwords:
            nodesToDel.extend(list(sum(G.edges(n), ())))

    nodesToDel = set(nodesToDel)

    g = G.subgraph([term for term, _ in pos.items() if term not in nodesToDel])
    nx.draw_networkx_labels(g, pos, font_size=40, alpha=0.9)
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='g',
                           alpha=0.3)

    plt.axis('off')
    plt.savefig(outfile % "-no-womenwords")

    plt.clf()

    # GRAPH 3

    for womanword in testwords:
        if womanword in G:
            nodesToKeep = set(list(sum(G.edges(womanword), ())))
            g = G.subgraph([term for term, _ in pos.items()
                            if term in nodesToKeep])
            nx.draw_networkx_labels(g, pos, font_size=40, alpha=0.9)
            nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='r',
                                   alpha=0.3)
            plt.axis('off')
            plt.savefig(outfile % ("-" + womanword))

            plt.clf()

            g = G.subgraph([term for term, _ in pos.items()
                            if term not in nodesToKeep])
            nx.draw_networkx_labels(g, pos, font_size=40, alpha=0.9)
            nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='g',
                                   alpha=0.3)
            plt.axis('off')
            plt.savefig(outfile % ("-no-" + womanword))

            plt.clf()


if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
outdir = "networks/dense/" + path.replace("/", "-").replace("-out", "").replace("books-genji-", "")
outfile = outdir + "-network_m-w%s-%s-%s-%sthresh.png" % (NUMTERMS, "%s", sys.argv[2], valthresh)

model = LdaModel.load(sys.argv[1])

graph_terms_to_topics(model, outfile)
