import json
import nli
from tf_shallow_neural_classifier import TfShallowNeuralClassifier
import os
import random
import tensorflow as tf
import itertools

hidden_dims = [50,100,200]
max_iters = [5,10,20]
vocab_dims= [50]
activations = [tf.nn.relu, tf.nn.tanh]
etas = [0.01, 0.001, 0.0001]
hyps = itertools.product(hidden_dims, max_iters, vocab_dims, activations, etas)
best_embeddings = ""
bestF1 = 0
best_params = None
for hyp in [[100,20, 50, tf.nn.relu, 0.001], [100,20, 50, tf.nn.relu, 0.001]]:
    hypname = ""
    for param in hyp:
        hypname += str(param)
    train, dev, test = nli.create_split(["allposWN"]*3,[None,None,None], disjoint=False)
    model = TfShallowNeuralClassifier(hidden_dim=hyp[0], max_iter=hyp[1], vocab_dim=hyp[2], hidden_activation=hyp[3], eta=hyp[4], embedding_dir="embeddings" + hypname, encoder=True)
    results = nli.encoder_experiment(train,test,model)
    if results["macro-F1"] > bestF1:
        best_embeddings = "embeddings" + hypname
        best_params = hyp
print(best_params)

ratios = [0.5, 0.25, 0.125, 0.06025, 0.030125]
for ratio in ratios:
    source = str(ratio) + "posWN"
    train, dev, test = nli.create_split([source + ".train", source + ".dev", source + ".test"],[None,None,None], disjoint=False)
    model = TfShallowNeuralClassifier(hidden_dim=hyp[0], max_iter=hyp[1], vocab_dim=hyp[2], hidden_activation=hyp[3], eta=hyp[4],embedding_dir=best_embeddings + ".npy", encoder=False)
    nli.decoder_experiment(train,test,model)
