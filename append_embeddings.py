import sys
import nli
import numpy as np
from nltk.corpus import wordnet as wn

root_dir = sys.argv[1]
contra_dir = sys.argv[2]
anto_dir = sys.argv[3]
new_dir = sys.argv[4]
wtoi = nli.word_to_ind()
contra_mat = np.load(contra_dir)
anto_mat = np.load(anto_dir)
with open(new_dir, "w", encoding="utf-8") as g:
    with open(root_dir, "r", encoding="utf-8") as f:
        for line in f.readlines():
            count +=1
            suffix = ""
            if len(wn.synsets(line.split()[0])) !=0:
                synset = wn.synsets(line.split()[0])[0].name()
                lemma =  str(wn.synsets(line.split()[0])[0].lemmas()[0])
            else:
                synset = ""
                lemma = ""
            if  synset in wtoi:
                for i in contra_mat[wtoi[synset], :]:
                    suffix += " " + str(i)
            else:
                for i in contra_mat[0, :]:
                    suffix += " 0"
            if lemma in wtoi:
                for i in anto_mat[wtoi[lemma], :]:
                    suffix += " " + str(i)
            else:
                for i in anto_mat[0, :]:
                    suffix += " 0"
            g.write(line + suffix + "\n")
