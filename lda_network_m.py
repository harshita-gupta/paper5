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

NUMTERMS = 50
valthresh = 0.0035

womenwords = set(['daughter', 'ladi', 'princess', 'wife', 'women', 'woman',
                  'mother', 'girl', 'sister'])

#############
# cf. http://nbviewer.ipython.org/github/sgsinclair/alta/blob/master/ipynb/TopicModelling.ipynb#Graphing-Topic-Terms

def graph_terms_to_topics(lda, outfile, num_terms=NUMTERMS):

    # create a new graph and size it
    G = nx.Graph()
    plt.figure(figsize=(50, 50))

    # generate the edges
    for i in range(0, lda.num_topics):
        # topicLabel = "topic " + str(i)
        terms = [term for term, val in lda.show_topic(i, num_terms) if val > valthresh]
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
    pos = nx.spring_layout(G, k=0.30, iterations=80000)

    g = G.subgraph([term for term, _ in pos.items()])
    nx.draw_networkx_labels(g, pos, font_size=25, alpha=0.9)

    # plot edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='g',
                           alpha=0.3)

    for n in G:
        if n in womenwords:
            nx.draw_networkx_edges(G, pos, edgelist=nx.edges(G, nbunch=n),
                                   edge_color='r', weight=1.2)

    print len(G)

    plt.axis('off')
    plt.savefig(outfile % "-with-women")

    plt.clf()


    ##########
    # GRAPH 2

    nodesToDel = []
    for n in G:
        if n in womenwords:
            # nx.draw_networkx_edges(G, pos, edgelist=nx.edges(G, nbunch=n),
            #                        edge_color='r', alpha=0.0)
            nodesToDel.extend(list(sum(G.edges(n), ())))

    nodesToDel = set(nodesToDel)

    for node in nodesToDel:
        G.remove_node(node)

    g = G.subgraph([term for term, _ in pos.items() if term not in nodesToDel])
    nx.draw_networkx_labels(g, pos, font_size=25, alpha=0.9)

    # plot edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='g',
                           alpha=0.3)


    print len(G)
    plt.axis('off')
    plt.savefig(outfile % "-no-women")

    # plt.show()



if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
outdir = "networks/dense/" + path.replace("/", "-").replace("-out", "").replace("books-genji-", "")
outfile = outdir + "-network_m-w%s-%s-w%s.png" % (NUMTERMS, "%s", sys.argv[2])

model = LdaModel.load(sys.argv[1])

graph_terms_to_topics(model, outfile)
