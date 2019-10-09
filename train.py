import json
import nli
from tf_shallow_neural_classifier import TfShallowNeuralClassifier
import os
import random

best_embeddings = ""
bestF1 = 0
for hyp in [[0], [1]]:
    hypname = ""
    for param in hyp:
        hypname += str(param)
    train, dev, test = nli.create_split(["allposWN"]*3,[None,None,None], disjoint=False)
    model = TfShallowNeuralClassifier(hidden_dim=2, max_iter=1, vocab_dim=2, embedding_dir="embeddings" + hypname, encoder=True)
    results = nli.encoder_experiment(train,test,model)
    if results["macro-F1"] > bestF1:
        best_embeddings = "embeddings" + hypname

ratios = [0.5, 0.25, 0.125, 0.06025, 0.030125]
for ratio in ratios:
    source = str(ratio) + "posWN"
    train, dev, test = nli.create_split([source + ".train", source + ".dev", source + ".test"],[None,None,None], disjoint=False)
    model = TfShallowNeuralClassifier(hidden_dim=2, max_iter=1, vocab_dim=2, embedding_dir=best_embeddings + ".npy", encoder=False)
    nli.decoder_experiment(train,test,model)
