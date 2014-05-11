from nlplearn import *
from operator import itemgetter
import nltk


class LanguageParser(BaseEstimator, TransformerMixin):

    '''Parse the sentences (tokenizing, tagging, chunk parsing)'''

    def __init(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        for doc in documents:
            if not 'tokens' in doc.ext:
                doc.ext['tokens'] = [nltk.word_tokenize(s) for s in doc.ext['sentences'] if s]
            if not 'tags' in doc.ext:
                doc.ext['tags'] = [nltk.pos_tag(t) for t in doc.ext['tokens'] if t]
            if not 'chunks' in doc.ext:
                doc.ext['chunks'] = [nltk.chunk.ne_chunk(t) for t in doc.ext['tags'] if t]
        return documents


class EntityNumberEstimator(HeadlineEstimator):
    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def predict(self, documents):
        return [max([(sum(1 for _ in sentence[1].subtrees()),sentence) for sentence in zip(doc.ext['sentences'], doc.ext['chunks']) ],key=itemgetter(0))[1][0] for doc in documents]
