from collections import defaultdict
import json
from nltk.tree import Tree
import numpy as np
import os
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


CONDITION_NAMES = [
    'edge_disjoint',
    'word_disjoint',
    'word_disjoint_balanced']

def word_to_ind():
    with open(os.path.join('data', 'wordtoind'), "r") as f:
        return json.loads(f.readline().strip())

def create_dataset(sourcename, size = None, restrictions = None):
    """ Create a dataset of balanced positive and negative examples from a raw list of positive examples.
    Parameters
    ----------
    sourcename : string
        The name of a JSON file containing a list of positive examples.
    destname : string
        The name of the file to write the dataset to
    size : int
        Size of the returned dataset
    restrictions : set
        A set of pairs of negative examples that will not occur in the produced dataset
    """
    with open(os.path.join("data", sourcename), "r") as f:
        target_pairs = json.loads(f.readline().strip())
    with open(os.path.join("data","allposWN")) as f:
        all_pairs = json.loads(f.readline().strip())
    with open(os.path.join("data","WNvocab")) as f:
        vocab = json.loads(f.readline().strip())
    random.shuffle(target_pairs)
    wtoi = word_to_ind()
    dataset = []
    all_pairs = {tuple(x) for x in all_pairs}
    count = 0
    for pair in target_pairs:
        count +=1
        if size is not None and count > size:
            break
        if count % 2 == 0:
            yield (wtoi[pair[0]], wtoi[pair[0]]), 1
        else:
            w1 = random.choice(vocab)
            w2 = random.choice(vocab)
            while (w1, w2) in all_pairs or (w1,w2) in restrictions:
                w1 = random.choice(vocab)
                w2 = random.choice(vocab)
            yield (wtoi[w1],wtoi[w2]), 0

def create_split(sourcenames, sizes,disjoint=True):
    """ Create multiple datasets for training, development, and testing.
    Parameters
    ----------
    sourcenames : list of strings
        A list of names of Json files containing lists of positive examples.
    destnames : list of strings
        A list of names to write the datasets to
    sizes : list of ints
        A list of sizes that the datasets should be
    """
    restrictions = set()
    generators = []
    sourcenames.reverse()
    sizes.reverse()
    for sourcename, size in zip(sourcenames,sizes):
        generators.append(lambda: create_dataset(sourcename, size, restrictions))
        if disjoint:
            for pair, label in generators[-1]:
                restrictions.append(pair)


    generators.reverse()
    return generators


def encoder_experiment(
        train_data,
        assess_data,
        model):
    """Train and evaluation code for the word-level entailment task.

    Parameters
    ----------
    train_data : list
    assess_data : list
    vector_func : function
        Any function mapping words in the vocab for `wordentail_data`
        to vector representations
    model : class with `fit` and `predict` methods

    Prints
    ------
    To standard ouput
        An sklearn classification report for all three splits.

    Returns
    -------
    dict with structure

        'model': the trained model
        'train_condition': train_condition
        'assess_condition': assess_condition
        'macro-F1': score for 'assess_condition'
        'vector_func': vector_func

    We pass 'vector_func' through to ensure alignment
    between these experiments and the bake-off evaluation.

    """
    model.fit(train_data)
    predictions = model.predict(assess_data)
    # Report:
    y = []
    for input, label in assess_data():
        y.append(label)
    print(classification_report(y, predictions))
    macrof1 = utils.safe_macro_f1(y, predictions)
    return {
        'model': model,
        'train_data': train_data,
        'assess_data': assess_data,
        'macro-F1': macrof1}

def decoder_experiment(
        train_data,
        assess_data,
        model):
    """Train and evaluation code for the word-level entailment task.

    Parameters
    ----------
    train_data : list
    assess_data : list
    vector_func : function
        Any function mapping words in the vocab for `wordentail_data`
        to vector representations
    model : class with `fit` and `predict` methods

    Prints
    ------
    To standard ouput
        An sklearn classification report for all three splits.

    Returns
    -------
    dict with structure

        'model': the trained model
        'train_condition': train_condition
        'assess_condition': assess_condition
        'macro-F1': score for 'assess_condition'
        'vector_func': vector_func

    We pass 'vector_func' through to ensure alignment
    between these experiments and the bake-off evaluation.

    """
    model.fit(train_data)
    predictions = model.predict(assess_data)
    # Report:
    y = []
    for input, label in assess_data():
        y.append(label)
    print(classification_report(y, predictions))
    macrof1 = utils.safe_macro_f1(y, predictions)
    return {
        'model': model,
        'train_data': train_data,
        'assess_data': assess_data,
        'macro-F1': macrof1}
