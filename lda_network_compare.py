#!/usr/bin/env python

# The python code in development for visualization the connections
# between topics.
# Harshita Gupta. Humanities Colloqium. Open-Ended Project 5. Spring 2017.

from gensim.models import LdaModel
import networkx as nx
import matplotlib
import cPickle as pickle
import sys
import os
matplotlib.use('TkAgg')
from copy import deepcopy
import matplotlib.pyplot as plt

fromPickle = True

NUMTERMS = 50
valthresh = 0.0040

washburn = "books/genji/washburn/out/ldamodel_overlap0_w1000_t15_p30_modified1.lda"
seidensticker = "books/genji/seidensticker/out/ldamodel_overlap0_w1000_t15_p30_modified1.lda"
tyler = "books/genji/tyler/out/ldamodel_overlap0_w1000_t15_p30_modified1.lda"
texts = [seidensticker, tyler, washburn]
graphs = []

womenwords = set(['daughter', 'ladi', 'princess', 'wife', 'women', 'woman',
                  'mother', 'girl', 'sister', 'mistress'])

mendependentwomen = set(['daughter', 'wife', 'mother', 'sister', 'mistress'])

singularwomen = womenwords - mendependentwomen

menwords = set(['son', 'prince', 'father', 'emperor', 'minist',
                'lord', 'man', 'husband', 'captain', 'men',
                'lordship'])

feelingthinking = set(['feel', 'think', 'felt', 'thought', 'tear',
                       'emot', 'die'])

performanceappearance = set(['perform', 'play', 'dress', 'robe', 'music',
                             'majesti'])

dutypalace = set(['palace', 'ceremoni', 'attend', 'rever', 'high', 'majesti'])


#############
def graph_terms_to_topics(lda, num_terms=NUMTERMS):

    # def save_clear(fmstr):
    #     plt.axis('off')
    #     plt.savefig(outfile % fmstr)
    #     plt.clf()

    G = nx.Graph()
    # plt.figure(figsize=(40, 40))

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
                        if term < term2:
                            G.add_edge(term, term2)
                        else:
                            G.add_edge(term2, term)

    return G

    # if fromPickle:
    #     with open((outfile % "positions-") + ".d", 'rb') as f:
    #         pos = pickle.load(f)
    # else:
    #     pos = nx.spring_layout(G, k=0.30, iterations=80000)
    #     with open((outfile % "positions-") + ".d", 'wb') as f:
    #         pickle.dump(pos, f)

    # def drawGraph(g, font, color1='m', color2='r', c3='b'):
    #     nx.draw_networkx_labels(g, pos, font_size=font, alpha=1.0)
    #     nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='g',
    #                            alpha=0.4)
    #     for n in g:
    #         if n in menwords:
    #             nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g, nbunch=n),
    #                                    edge_color='b', weight=1.0)
    #         if n in singularwomen:
    #             nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g, nbunch=n),
    #                                    edge_color=color1, weight=1.0)
    #         if n in mendependentwomen:
    #             nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g, nbunch=n),
    #                                    edge_color=color2, weight=1.0)
    #             # medges = nx.edges(g, nbunch=n)
    #             # medges = [(n1, n2) for (n1, n2) in medges if n2 in womenwords]
    #             # nx.draw_networkx_edges(g, pos, edgelist=medges,
    #             #                        edge_color='y', weight=1.0)

    # # GRAPH WITH EVERYTHING
    # g = G.subgraph([term for term, _ in pos.items()])
    # drawGraph(g, 25, 'r', 'r', 'r')
    # save_clear("full")
    # drawGraph(g, 25)
    # save_clear("full-split")

    # def removespecificcategory(catname, twords):
    #     nodesToDel = []
    #     for n in G:
    #         if n in twords:
    #             nodesToDel.extend(list(sum(G.edges(n), ())))
    #             # nodesToDel.append(n)

    #     nodesToDel = set(nodesToDel)

    #     g = G.subgraph([t for t, _ in pos.items() if t not in nodesToDel])
    #     drawGraph(g, 30)

    #     save_clear(("no-" + catname))


g1 = graph_terms_to_topics(LdaModel.load(texts[0]))
graphs.append(g1)
overlapedges = set(g1.edges())
c_nodes = set([n for n in list(sum(overlapedges, ()))])
individualedges = [deepcopy(overlapedges)]
individualnodes = [deepcopy(c_nodes)]

for text in texts[1:]:
    g = graph_terms_to_topics(LdaModel.load(text))
    graphs.append(g)
    edges = set(g.edges())
    nodes = set([n for n in list(sum(edges, ()))])
    individualedges.append(edges)
    individualnodes.append(nodes)
    overlapedges = overlapedges.intersection(edges)
    c_nodes = c_nodes.intersection(nodes)


def graph(edges, file):
    G = nx.Graph()
    plt.figure(figsize=(40, 40))
    for (n1, n2) in edges:
        G.add_edge(n1, n2)
    pos = nx.spring_layout(G, k=0.30, iterations=8000)

    def drawGraph(g, font, color1='m', color2='r', c3='b'):
        nx.draw_networkx_labels(g, pos, font_size=font, alpha=1.0)
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='g',
                               alpha=0.4)
        for n in g:
            if n in menwords:
                nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g, nbunch=n),
                                       edge_color='b', weight=1.0)
            if n in singularwomen:
                nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g, nbunch=n),
                                       edge_color=color1, weight=1.0)
            if n in mendependentwomen:
                nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g, nbunch=n),
                                       edge_color=color2, weight=1.0)

    drawGraph(G, 30)
    plt.axis('off')
    plt.savefig(file)
    plt.clf()
    return G


# graph(overlapedges, "overlap.png")

num = 1
for individualedge in individualedges:
    # graph(individualedge.difference(overlapedges), str(num) + ".png")
    num += 1

print "common nodes"
print c_nodes
print ""

print "graph's unique nodes:"
for individualnode, g in zip(individualnodes, graphs):
    print sorted(individualnode.difference(c_nodes), key=lambda x: g.degree(x))
    print ""

print "intersection seidensticker tyler"
print individualnodes[0].intersection(individualnodes[1]).difference(c_nodes)
print ""

print "intersection tyler washburn"
print individualnodes[1].intersection(individualnodes[2]).difference(c_nodes)
print ""

print "intersection washburn seidensticker"
print individualnodes[2].intersection(individualnodes[0]).difference(c_nodes)
print ""
