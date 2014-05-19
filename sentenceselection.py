from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import copy


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
