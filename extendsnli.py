from nltk.corpus import wordnet as wn
import random
import copy
import json
from string import punctuation

def find_match_word(example):
    prem = example["sentence1"].split()
    hyp = example["sentence2"].split()
    i = 0
    while True:
        if prem[i] != hyp[i]:
            break
        i += 1
    word1 = prem[i]
    word2 = hyp[i]
    if word1[-1] in punctuation:
        word1 = word1[:-1]
    if word2[-1] in punctuation:
        word2 = word2[:-1]
    if word1 in ["a", "an", "A", "An"]:
        word1 = prem[i+1]
        word2 = hyp[i+1]
    return (word1,word2)

def inside(pair, contra, anto):
    first = pair[0]
    second = pair[1]
    firstsyn = ""
    secondsyn = ""
    firstlem= ""
    secondlem= ""
    if len(wn.synsets(first)) != 0:
        firstsyn = wn.synsets(first)[0].name()
        firstlem = str(wn.synsets(first)[0].lemmas()[0])
    if len(wn.synsets(second)) != 0:
        secondsyn = wn.synsets(second)[0].name()
        secondlem = str(wn.synsets(second)[0].lemmas()[0])
    print(firstsyn, secondsyn, firstlem, secondlem)
    if (firstsyn, secondsyn) in contra or (firstlem, secondlem) in anto:
        return True
    return False

def valid_contra(word, contra):
    if len(wn.synsets(word)) ==0:
        return None
    for pair in contra:
        if wn.synsets(word)[0].name() == pair[0] and wn.synset(pair[1]).lemmas()[0].name() != word and not "_" in wn.synset(pair[1]).lemmas()[0].name():
            return wn.synset(pair[1]).lemmas()[0].name()
    return None

def valid_syn(word):
    if len(wn.synsets(word)) <2:
        return None
    new = word
    while new == word:
        new = wn.synsets(word)[1 + random.randint(0, len(wn.synsets(word)) - 2)].lemmas()[0].name()
    if "_" in new:
        return None
    return new


with open("data\\contraposWN", "r") as f:
    contra = json.loads(f.readline().strip())
with open("data\\antoposWN", "r") as f:
    anto = json.loads(f.readline().strip())

if False:
    temp = set()
    for pair in contra:
        temp.add((pair[0], pair[1]))
    contra = temp
    temp = set()
    for pair in anto:
        temp.add((pair[0], pair[1]))
    anto = temp

size = 10
count = {"contradiction":0, "entailment":0, "neutral":0}
new = []
with open("snli_1.0\\snli_1.0_train.jsonl", "r") as f:
    for line in f:
        example = json.loads(line.strip())
        if example["gold_label"] == "contradiction" and count["contradiction"] < size:
            new_example = dict()
            prem = example["sentence1"]
            leng = len(prem.split())
            start = random.randint(0, leng-1)
            hyp = copy.copy(prem)
            newword= None
            for i in range(leng):
                newword = valid_contra(prem.split()[(start + i)%leng], contra)
                if newword is not None and newword !=prem.split()[(start + i)%leng] and prem.split()[(start + i)%leng] not in ["a", "an", "A", "An"]:
                    hyp = [x if j !=(start+i)%leng else newword for j,x in enumerate(hyp.split())]
                    break
            if hyp != prem:
                new_example["sentence1"] = " ".join(prem.split())
                new_example["sentence2"] = " ".join(hyp)
                new_example["gold_label"] = "contradiction"
                new.append(new_example)
                count["contradiction"] +=1
        if example["gold_label"] == "entailment" and count["entailment"] < size:
            new_example = dict()
            prem = example["sentence1"]
            leng = len(prem.split())
            start = random.randint(0, leng-1)
            hyp = copy.copy(prem)
            newword= None
            for i in range(leng):
                newword = valid_syn(prem.split()[(start + i)%leng])
                if newword is not None and newword != prem.split()[(start + i)%leng] and prem.split()[(start + i)%leng] not in ["a", "an", "A", "An"]:
                    hyp = [x if j !=(start+i)%leng else newword for j,x in enumerate(hyp.split())]
                    break
            if hyp != prem:
                new_example["sentence1"] = " ".join(prem.split())
                new_example["sentence2"] = " ".join(hyp)
                new_example["gold_label"] = "entailment"
                new.append(new_example)
                count["entailment"] +=1
        if example["gold_label"] == "neutral" and count["neutral"] < size:
            new.append(example)
            count["neutral"] +=1
        print(count)
        print(new)

print(new)



result = dict()
with open("dataset.jsonl", "r") as f:
    good = 0
    total = 0
    for line in f:
        example = json.loads(line.strip())
        if example["gold_label"] != "contradiction":
            continue
        cat = example["category"]
        if cat not in result:
            result[cat] = {"total":0, "good":0}
        result[cat]["total"]+=1
        word_pair = find_match_word(example)
        temp = inside(word_pair,contra, anto)
        print(example["sentence1"], example["sentence2"])
        print(word_pair, temp)
        if temp:
            result[cat]["good"]+=1
good = 0
total = 0
for cat in result:
    good += result[cat]["good"]
    total += result[cat]["total"]
    print(cat, result[cat]["total"], result[cat]["good"], result[cat]["good"]/result[cat]["total"])
print(good, total, good/total)
