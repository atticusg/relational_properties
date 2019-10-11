import re
import pandas
import os
import json
from nltk.corpus import wordnet as wn
from tqdm import tqdm
try:
    wn.all_synsets
except LookupError as e:
    import nltk
    nltk.download('wordnet')

# make sure each edge is included only once
edges = set()
base_edges = set()
words = set()
contra_edges = set()
for synset in tqdm(wn.all_synsets(pos='n')):
    # write the transitive closure of all hypernyms of a synset to file
    for hyper in synset.hypernyms():
        for grandhyper in hyper.hypernyms():
            for aunt in grandhyper.hyponyms():
                contra_edges.add((synset.name(), aunt.name()))
                contra_edges.add((aunt.name(), synset.name()))
                for cousin in aunt.hyponyms():
                    contra_edges.add((synset.name(), cousin.name()))
                    contra_edges.add((cousin.name(), synset.name()))
    if (synset.name(), synset.name()) in contra_edges:
        contra_edges.remove((synset.name(), synset.name()))
    for hyper in synset.hypernyms():
        if (synset.name(), hyper.name()) in contra_edges:
            contra_edges.remove((synset.name(), hyper.name()))
        if (hyper.name(), synset.name()) in contra_edges:
            contra_edges.remove((hyper.name(),synset.name()))
        for grandhyper in hyper.hypernyms():
            if (synset.name(), grandhyper.name()) in contra_edges:
                contra_edges.remove((synset.name(), grandhyper.name()))
            if (grandhyper.name(), synset.name()) in contra_edges:
                contra_edges.remove((grandhyper.name(),synset.name()))
    for hyper in synset.hypernyms():
        base_edges.add((synset.name(), hyper.name()))
    for hyper in synset.closure(lambda s: s.hypernyms()):
        edges.add((synset.name(), hyper.name()))
    words.add(synset.name())

word_to_ind = dict()
ind = 0
for word in words:
    word_to_ind[word] = ind
    ind +=1

print(len(contra_edges))
with open(os.path.join('data', 'wordtoind'), "w") as f:
    f.write(json.dumps(word_to_ind))
with open(os.path.join('data', 'WNvocab'), "w") as f:
    f.write(json.dumps(list(words)))
with open(os.path.join('data', 'allposWN'), "w") as f:
    f.write(json.dumps(list(edges)))
with open(os.path.join('data', 'contraallposWN'), "w") as f:
    f.write(json.dumps(list(contra_edges)))
ratios = [0.5, 0.25, 0.125, 0.06025, 0.030125]
l = len(edges)
for prefix in ["", "contra"]:
    for ratio in ratios:
        if prefix == "contra":
            edges = contra_edges
        with open(os.path.join('data', str(ratio) + prefix +'posWN.train'), "w") as f:
            f.write(json.dumps(list(edges)[:int(ratio*l)]))
        with open(os.path.join('data', str(ratio) + prefix +'posWN.dev'), "w") as f:
            f.write(json.dumps(list(edges)[int(ratio*l):int(((1-ratio)*0.5 + ratio)*l)]))
        with open(os.path.join('data', str(ratio) + prefix +'posWN.test'), "w") as f:
            f.write(json.dumps(list(edges)[int(((1-ratio)*0.5 + ratio)*l):]))
with open(os.path.join('data', 'transbaseWN'), "w") as f:
    f.write(json.dumps(list(base_edges)))
with open(os.path.join('data', 'transdiffWN'), "w") as f:
    f.write(json.dumps(list(edges.difference(base_edges))))
