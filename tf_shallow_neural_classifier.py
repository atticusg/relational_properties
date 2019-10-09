import tensorflow as tf
from tf_model_base import TfModelBase
import numpy as np
import nli

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TfShallowNeuralClassifier(TfModelBase):
    def __init__(self, **kwargs):
        super(TfShallowNeuralClassifier, self).__init__(**kwargs)

    def fit(self, X):
        word_to_ind = nli.word_to_ind()
        self.vocab_size = len(word_to_ind.keys())
        self.classes_ = [0,1]
        self.n_classes_ = 2
        self.estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.model_dir)
        input_fn = lambda: self._train_input_fn(X)
        self.estimator.train(input_fn)
        return self

    def _train_input_fn(self, X):
        dataset = tf.data.Dataset.from_generator(generator=X, output_types=(tf.int32,tf.int32), output_shapes = ((2,), ()))
        dataset = (dataset
                    .repeat(self.max_iter)
                    .batch(self.batch_size))
        return dataset

    def _test_input_fn(self, X):
        dataset = tf.data.Dataset.from_generator(generator=X, output_types=(tf.int32,tf.int32), output_shapes = ((2,), ()))
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _model_fn(self, features, labels, mode):
        if self.encoder:
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size,self.vocab_dim], -1.0, 1.0), name="Embed")
            X1 = tf.nn.embedding_lookup(embeddings, features[:,0])
            X2 = tf.nn.embedding_lookup(embeddings, features[:,1])
            features = tf.concat([X1,X2], axis=1)
        else:
            embeddings = tf.Variable(np.load(self.embedding_dir), trainable=False)
            X1 = tf.nn.embedding_lookup(embeddings, features[:,0])
            X2 = tf.nn.embedding_lookup(embeddings, features[:,1])
            features = tf.concat([X1,X2], axis=1)
        # Graph:
        hidden = tf.layers.dense(
            features,
            self.batch_size,
            activation=self.hidden_activation)
        logits = tf.layers.dense(
            hidden,
            self.n_classes_)
        # Predictions:
        preds = tf.argmax(logits, axis=-1)
        # Predicting:
        if mode == tf.estimator.ModeKeys.PREDICT:
            if self.encoder:
                    np.save(self.embedding_dir,self.estimator.get_variable_value(embeddings.name))
            proba = tf.nn.softmax(logits)
            results = {'proba': proba, 'pred': preds}
            return tf.estimator.EstimatorSpec(mode, predictions=results)
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(
                logits=logits, labels=labels)
            metrics = {
                'accuracy': tf.metrics.accuracy(labels, preds)
            }
            # Evaluation mode to enable early stopping based on metrics:
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            # Training:
            elif mode == tf.estimator.ModeKeys.TRAIN:
                global_step = tf.train.get_or_create_global_step()
                train_op = tf.train.AdamOptimizer(self.eta).minimize(
                    loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    def predict_proba(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [x['proba'] for x in self.estimator.predict(input_fn)]

    def predict(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [self.classes_[x['pred']] for x in self.estimator.predict(input_fn)]


def simple_example():
    """Assess on the digits dataset."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    mod = TfShallowNeuralClassifier()

    print(mod)

    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)

    print(classification_report(y_test, predictions))

    return accuracy_score(y_test, predictions)


if __name__ == '__main__':
   simple_example()
