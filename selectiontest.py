from nlpio import *
from trimming import *
from nlplearn import *
from sentenceselection import *
from sklearn.pipeline import Pipeline
import pickle
import os.path


if __name__ == '__main__':

    filename = 'testset'
    if os.path.isfile('.'.join([filename, 'pk'])):
        # Already parsed by stanford corenlp
        documents = pickle.load(open('.'.join([filename, 'pk'])))
    else:
        # Brace yourself
        documents = loadDocumentsFromFile('.'.join([filename, 'txt']))

    ls = LinearSelector()
    ls.addFeature(IsFirst())
    ls.addFeature(Length())
    ls.addFeature(WordCoverage(stop_words='english'))
    ls.addFeature(PositionInDocument())
    ls.addFeature(NamedEntityCount())
    ls.addFeature(Constant())

    pipeline = Pipeline([
        ('clean', SimpleTextCleaner()),
        ('split', SentenceSplitter()),
        ('parse', StanfordParser()),
        ('select', ls),
        ])

    pipeline.fit(documents)

    scorer = RougeScorer()

    print "ROUGE score: %f" % scorer(pipeline, documents)
