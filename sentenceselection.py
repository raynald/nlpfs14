from nlpio import evaluateRouge
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import numpy as np
import logging


class FirstSentenceSelector(BaseEstimator, TransformerMixin):

    '''Selects the first sentence of each article.'''

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        documents_copy = []
        for doc in documents:
            copy_doc = copy.deepcopy(doc)
            copy_doc.ext['sentences'] = [copy_doc.ext['sentences'][0]]
            copy_doc.ext['article'] = [copy_doc.ext['article'][0]]
            documents_copy.append(copy_doc)
        return documents_copy

    def predict(self, documents):
        copy_documents = self.transform(documents)
        output = []
        for doc in copy_documents:
            output.append(doc.ext['sentences'][0])
        return output


class Feature:
    ''' Base class for features used by LinearSelector.'''
    def __init__(self):
        pass

    def evaluate(self, document):
        pass

    def prepare(self, document):
        pass


class IsFirst(Feature):
    ''' 1 if first sentence, 0 otherwise.'''
    def evaluate(self, document):
        output = np.zeros((len(document.ext['article']),))
        output[0] = 1.0
        return output


class PositionInDocument(Feature):
    ''' Yields the normalized position in the document.'''
    def evaluate(self, documents):
        output = (np.array(range(len(documents.ext['article'])))
                  / float(len(documents.ext['article'])))
        return output


class Constant(Feature):
    ''' Constant feature.'''
    def evaluate(self, document):
        output = np.ones((len(document.ext['article']),))
        return output


class Length(Feature):
    ''' The length of the sentence.'''
    def evaluate(self, document):
        output = []
        for sentence in document.ext['article']:
            output.append(float(len(sentence['text'])))
        return np.array(output)


class WordCoverage(Feature):
    ''' A estimation of the coverage of a sentence by summing the components of
        its tfidf representation.'''
    def __init__(self, **kwargs):
        self.tfidf = TfidfVectorizer(**kwargs)

    def evaluate(self, document):
        sentences = []
        for sentence in document.ext['article']:
            sentences.append(" ".join([word[1]['Lemma'] for word in
                                       sentence['words']]))
        bow = self.tfidf.fit_transform(sentences)
        return bow.sum(axis=1)


class NamedEntityCount(Feature):
    ''' Counts the number of named entities per sentence.'''
    def evaluate(self, document):
        output = []
        for sentence in document.ext['article']:
            count = 0
            last_was_entity = False
            for word in sentence['words']:
                if word[1]['NamedEntityTag'] != 'O':
                    if not last_was_entity:
                        count += 1
                    last_was_entity = True
                else:
                    last_was_entity = False
            output.append(float(count))
        return np.array(output)


class LinearSelector(BaseEstimator, TransformerMixin):

    '''Selects a sentence based on its score (weighted sum of features).'''

    def __init__(self, n_learning_iterations=5, scorer_func=evaluateRouge,
                 regularizer=1.0, learning_rate=1.0, variant='ROUGE-1',
                 metric='F'):
        self.features = []
        self.weights = []
        self.n_learning_iterations = n_learning_iterations
        self.scorer_func = scorer_func
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.variant = variant
        self.metric = metric

        self.covariance_matrix = None
        self.inv_covariance_matrix = None
        self.feedback = None
        self.last_feature_values = None

    def addFeature(self, feature, weight=1.0):
        self.features.append(feature)
        self.weights.append(weight)

    def transform(self, documents):
        documents_copy = []
        for doc in documents:
            copy_doc = copy.deepcopy(doc)
            score = np.zeros((len(copy_doc.ext['article']), 1))
            for weight, feature in zip(self.weights.flatten(), self.features):
                score += weight * np.reshape(feature.evaluate(copy_doc),
                                           (len(copy_doc.ext['article']), 1))
            best = np.argmax(score)
            copy_doc.ext['sentences'] = [copy_doc.ext['sentences'][best]]
            copy_doc.ext['article'] = [copy_doc.ext['article'][best]]
            documents_copy.append(copy_doc)
        return documents_copy

    def predict(self, documents):
        copy_documents = self.transform(documents)
        output = []
        for doc in copy_documents:
            output.append(doc.ext['article'][0]['text'])
        return output

    def fit(self, documents, y=None):
        ''' An online linear algorithm that tries to approximate the rouge
            score as weights * features.
        '''

        self.covariance_matrix = self.regularizer*np.eye(len(self.features))
        self.inv_covariance_matrix = (np.eye(len(self.features))
                                      / self.regularizer)
        self.feedback = np.zeros((len(self.features), 1))
        self.weights = np.array(self.weights)

        score = None
        for t in xrange(self.n_learning_iterations):
            logging.info('Learning iteration %i/%i...', t+1,
                         self.n_learning_iterations)
            for doc in documents:
                suggested = self.iterationStep_(t, score, doc)
                score = (self.scorer_func([doc], [suggested])
                         [self.variant][self.metric][0])

        return self

    def iterationStep_(self, t, score, document):

        if not score is None:
            self.covariance_matrix += np.dot(self.last_feature_values,
                                             self.last_feature_values.T)
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
            self.feedback += score * self.last_feature_values
            self.weights = np.dot(self.inv_covariance_matrix, self.feedback)

        feature_values = np.zeros((len(self.features),
                                   len(document.ext['article'])))
        for i, feature in enumerate(self.features):
            feature_values[i, :] = feature.evaluate(document).flatten()

        estimates = np.dot(feature_values.T, np.reshape(self.weights,
                           (len(self.weights), 1)))
        estimates_variance = np.zeros((len(document.ext['article']), 1))
        for j in xrange(len(document.ext['article'])):
            estimates_variance[j] = self.learning_rate * np.sqrt(
                np.dot(np.dot(feature_values[:, j].T,
                       self.inv_covariance_matrix), feature_values[:, j]))

        suggested_index = np.argmax(estimates + estimates_variance)
        self.last_feature_values = np.reshape(feature_values[:,
                                                             suggested_index],
                                              (feature_values.shape[0], 1))
        logging.debug('Selected sentence %i', suggested_index)
        return document.ext['article'][suggested_index]['text']
