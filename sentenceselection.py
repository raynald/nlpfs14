from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import heapq


class FirstSentenceSelector(BaseEstimator):

    '''Selects the best compression of the first sentence inferior in length
    to max_length, after ViterbiSentenceCompressor.'''

    def __init__(self, max_length=75):
        self.max_length = max_length

    def fit(self, documents, y=None):
        return self

    def predict(self, documents):
        output = []
        for doc in documents:
            q = [(-s[1], s[0]) for s in doc.ext['compressed_sentences'][0]]
            heapq.heapify(q)
            top = heapq.heappop(q)
            while (len(q) > 0 and (not top[1] is None) and
                    len(top[1]) > self.max_length):
                top = heapq.heappop(q)
            if top[1] is None or len(top[1]) > self.max_length:
                output.append('')
            else:
                output.append(top[1])
        return output
