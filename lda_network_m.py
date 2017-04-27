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
import matplotlib.pyplot as plt

fromPickle = True

NUMTERMS = 50
valthresh = 0.0040

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
def graph_terms_to_topics(lda, outfile, num_terms=NUMTERMS):

    def save_clear(fmstr):
        plt.axis('off')
        plt.savefig(outfile % fmstr)
        plt.clf()

    G = nx.Graph()
    plt.figure(figsize=(40, 40))

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
                        G.add_edge(term, term2, color="red")

    if fromPickle:
        with open((outfile % "positions-") + ".d", 'rb') as f:
            pos = pickle.load(f)
    else:
        pos = nx.spring_layout(G, k=0.30, iterations=80000)
        with open((outfile % "positions-") + ".d", 'wb') as f:
            pickle.dump(pos, f)

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
                # medges = nx.edges(g, nbunch=n)
                # medges = [(n1, n2) for (n1, n2) in medges if n2 in womenwords]
                # nx.draw_networkx_edges(g, pos, edgelist=medges,
                #                        edge_color='y', weight=1.0)

    # GRAPH WITH EVERYTHING
    g = G.subgraph([term for term, _ in pos.items()])
    drawGraph(g, 25, 'r', 'r', 'r')
    save_clear("full")
    drawGraph(g, 25)
    save_clear("full-split")

    def removespecificcategory(catname, twords):
        nodesToDel = []
        for n in G:
            if n in twords:
                nodesToDel.extend(list(sum(G.edges(n), ())))
                # nodesToDel.append(n)

        nodesToDel = set(nodesToDel)

        g = G.subgraph([t for t, _ in pos.items() if t not in nodesToDel])
        drawGraph(g, 30)

        save_clear(("no-" + catname))

    def removeindivwordscat(twords):
        for testword in twords:
            if testword in G:
                nodesToKeep = set(list(sum(G.edges(testword), ())))
                g = G.subgraph([term for term, _ in pos.items()
                                if term in nodesToKeep])
                drawGraph(g, 30)

                save_clear(testword)

                g = G.subgraph([term for term, _ in pos.items()
                                if term not in nodesToKeep])
                # g = G.subgraph([term for term, _ in pos.items()
                                # if term != testword])
                drawGraph(g, 30)

                save_clear(("no-" + testword))

    # removespecificcategory("womenwords", womenwords)
    # removeindivwordscat(womenwords)
    # removespecificcategory("feelingthinking", feelingthinking)
    # removeindivwordscat(feelingthinking)
    # removespecificcategory("mendependentwomen", mendependentwomen)
    removespecificcategory("menwords", menwords)
    removeindivwordscat(menwords)
    # removespecificcategory("performanceappearance", performanceappearance)
    # removeindivwordscat(performanceappearance)
    # removespecificcategory("dutypalace", dutypalace)
    # removeindivwordscat(dutypalace)


if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]
outdir = "networks/dense/" + path.replace("/", "-").replace("-out", "").replace("books-genji-", "")
outfile = outdir + "-network_m-w%s-%s-%s-%sthresh.png" % (NUMTERMS, "%s", sys.argv[2], valthresh)

model = LdaModel.load(sys.argv[1])

graph_terms_to_topics(model, outfile)
