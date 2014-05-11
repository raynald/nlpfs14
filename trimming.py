from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from nlpio import stanfordParse
from math import log
import numpy as np
import logging
import heapq


class StanfordParser(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        logging.info("Starting documents parsing with StanfordCoreNLP...")
        for i, doc in enumerate(documents):
            processed = False
            if not 'article' in doc.ext.keys():
                processed = True
                doc.ext['article'] = []
                for sentence in doc.ext['sentences']:
                    doc.ext['article'].extend(
                        stanfordParse(sentence)['sentences'])
            if not 'models' in doc.ext.keys():
                processed = True
                doc.ext['models'] = []
                for model in doc.models:
                    doc.ext['models'].extend(
                        stanfordParse(model)['sentences'])
            if processed:
                logging.info("Processed document %i/%i" % (i+1,
                                                           len(documents)))
            else:
                logging.info("Document %i/%i was already processed" %
                             (i+1, len(documents)))
        return documents


class SentenceCompressor(BaseEstimator, TransformerMixin):

    def __init__(self, max_n_words=25, tags_importance=0.7):
        self.max_n_words = max_n_words
        self.tags_importance = tags_importance

        self.headlinese = dict()  # bigrams
        self.headlinese_tags = dict()  # bigrams
        self.headlinese_unary = dict()  # unary
        self.headlinese_tags_unary = dict()  # unary
        self.english = dict()  # unary
        self.english_tags = dict()  # unary

        self.markers = {
            'begin': '$B$',
            'end': '$E$',
            'unknown': '$U$'
        }

    def fit(self, documents, y=None):
        ''' Learn headlinese unigram and bigram probabilities from model
            summaries and English unigram probabilities from the content of
            the articles. It requires that the documents have been parsed by
            StanfordParser beforehand.
            The behavior is a bit different from the usual sklearn transformers
            in the sense that it will not refit if it has already been fitted.
        '''
        if len(self.headlinese) > 0:
            logging.info('Already fitted.')
            return self
        logging.info('Starting learning language from documents...')
        for i, doc in enumerate(documents):
            # Headlinese (coded with bigrams)
            for model in doc.ext['models']:
                lastLemma = self.markers['begin']
                lastTag = self.markers['end']
                for word in model['words']:
                    # For words
                    lemma = word[1]['Lemma']
                    if not lastLemma in self.headlinese:
                        self.headlinese[lastLemma] = dict()
                        self.headlinese[lastLemma][lemma] = 1
                    elif not lemma in self.headlinese[lastLemma]:
                        self.headlinese[lastLemma][lemma] = 1
                    else:
                        self.headlinese[lastLemma][lemma] += 1
                    if not lemma in self.headlinese_unary:
                        self.headlinese_unary[lemma] = 1
                    else:
                        self.headlinese_unary[lemma] += 1
                    lastLemma = lemma
                    # For tags
                    tag = word[1]['PartOfSpeech']
                    if not lastTag in self.headlinese_tags:
                        self.headlinese_tags[lastTag] = dict()
                        self.headlinese_tags[lastTag][tag] = 1
                    elif not tag in self.headlinese_tags[lastTag]:
                        self.headlinese_tags[lastTag][tag] = 1
                    else:
                        self.headlinese_tags[lastTag][tag] += 1
                    if not tag in self.headlinese_tags_unary:
                        self.headlinese_tags_unary[tag] = 1
                    else:
                        self.headlinese_tags_unary[tag] += 1
                    lastTag = tag
                # For words
                end_marker = self.markers['end']
                if not lastLemma in self.headlinese:
                    self.headlinese[lastLemma] = dict()
                    self.headlinese[lastLemma][end_marker] = 1
                elif not end_marker in self.headlinese[lastLemma]:
                    self.headlinese[lastLemma][end_marker] = 1
                else:
                    self.headlinese[lastLemma][end_marker] += 1
                # For tags
                if not lastTag in self.headlinese_tags:
                    self.headlinese_tags[lastTag] = dict()
                    self.headlinese_tags[lastTag][end_marker] = 1
                elif not end_marker in self.headlinese_tags[lastTag]:
                    self.headlinese_tags[lastTag][end_marker] = 1
                else:
                    self.headlinese_tags[lastTag][end_marker] += 1
            # English (unary probabilities)
            for sentence in doc.ext['article']:
                for word in sentence['words']:
                    # For words
                    lemma = word[1]['Lemma']
                    if not lemma in self.english:
                        self.english[lemma] = 1
                    else:
                        self.english[lemma] += 1
                    # For tags
                    tag = word[1]['PartOfSpeech']
                    if not tag in self.english_tags:
                        self.english_tags[tag] = 1
                    else:
                        self.english_tags[tag] += 1
            logging.info("Processed document %i/%i" % (i+1, len(documents)))
        # Normalizing the probabilities
        logging.info("Normalizing probabilities...")
        self.normalize_()
        return self

    def normalizeDict_(self, d):
        n_once = sum([1 if v==1 else 0 for v in d.values()])
        t = sum(d.values())
        unknown_probability = n_once/float(t)
        if unknown_probability == 1.0:
            unknown_probability = 0.99
        elif unknown_probability == 0.0:
            unknown_probability = 0.01
        t /= 1.0 - unknown_probability
        for k, v in d.items():
            d[k] /= t
        d[self.markers['unknown']] = unknown_probability

        return d

    def normalize_(self):
        # Headlinese
        for d in self.headlinese.values():
            self.normalizeDict_(d)
        # Headlinese tags
        for d in self.headlinese_tags.values():
            self.normalizeDict_(d)
        # English
        self.normalizeDict_(self.english)
        # English tags
        self.normalizeDict_(self.english_tags)
        # Headlinese unary
        self.normalizeDict_(self.headlinese_unary)
        # Headlinese tags unary
        self.normalizeDict_(self.headlinese_tags_unary)

    def createWord_(self, word, tag):
        ''' Create a fake word in StanfordCoreNLP style.'''
        return (word, {'Lemma': word, 'PartOfSpeech': tag})

    def jointProbability_(self, sequence, last_word, next_word,
                          starting_point, end_point):
        ''' Compute the probability that next_word follows last_word in the
            headline sequence given last_word and all the words in between in
            the to-be-compressed sentence. starting_point should be the index
            of the word after last_word, and end_point the index of next_word.
        '''
        unknown_marker = self.markers['unknown']
        probability_words = 0.0
        probability_tags = 0.0
        # Words that are not in the headline sequence
        for i in xrange(starting_point, end_point):
            # For words
            lemma = sequence[i][1]['Lemma']
            if lemma in self.english:
                probability_words += log(self.english[lemma])
            else:
                probability_words += log(self.english[unknown_marker])
            # For tags
            tag = sequence[i][1]['PartOfSpeech']
            if tag in self.english_tags:
                probability_tags += log(self.english_tags[tag])
            else:
                probability_tags += log(self.english_tags[unknown_marker])
        # Headline sequence
        # For words
        last_lemma = last_word[1]['Lemma']
        next_lemma = next_word[1]['Lemma']
        if not last_lemma in self.headlinese:
            if not next_lemma in self.headlinese_unary:
                probability_words += log(self.headlinese_unary[unknown_marker])
            else:
                probability_words += log(self.headlinese_unary[next_lemma])
        elif not next_lemma in self.headlinese[last_lemma]:
            probability_words += log(self.headlinese[last_lemma]
                                     [unknown_marker])
        else:
            probability_words += log(self.headlinese[last_lemma][next_lemma])
        # For tags
        last_tag = last_word[1]['PartOfSpeech']
        next_tag = next_word[1]['PartOfSpeech']
        if not last_tag in self.headlinese_tags:
            if not next_tag in self.headlinese_tags_unary:
                probability_tags += log(
                    self.headlinese_tags_unary[unknown_marker])
            else:
                probability_tags += log(self.headlinese_tags_unary[next_tag])
        elif not next_tag in self.headlinese_tags[last_tag]:
            probability_tags += log(self.headlinese_tags[last_tag]
                                    [unknown_marker])
        else:
            probability_tags += log(self.headlinese_tags[last_tag][next_tag])

        return (self.tags_importance*probability_tags +
                (1-self.tags_importance)*probability_words)

    def backtrace_(self, sequence, backtrace, index, position):
        ''' Backtrace the sentence obtained in the Viterbi algorithm.'''
        if backtrace[position][index] == -1:
            return sequence[position][0]
        else:
            return "%s %s" % (self.backtrace_(sequence, backtrace, index-1,
                                              backtrace[position][index]),
                              sequence[position][0])

    def transform(self, documents):
        '''Infer the best sentence for all lengths up to max_n_words.'''
        for doc in documents:
            doc.ext['compressed_sentences'] = []
            for sentence in doc.ext['article']:
                doc.ext['compressed_sentences'].append([])
                backtrace = -1*np.ones((len(sentence['words']),
                                        self.max_n_words),
                                       dtype=int)
                probability = -np.Infinity*np.ones((len(sentence['words']),
                                                    self.max_n_words),
                                                   dtype=float)
                # Initialization
                start_marker = self.createWord_(self.markers['begin'],
                                                self.markers['begin'])
                for i in xrange(len(sentence['words'])):
                    probability[i][0] = self.jointProbability_(
                        sentence['words'], start_marker, sentence['words'][i],
                        0, i)
                # Main loop
                for index in xrange(1, self.max_n_words):
                    for next_position in xrange(len(sentence['words'])):
                        for last_position in xrange(next_position):
                            if (probability[last_position][index-1] ==
                                    -np.Infinity):
                                continue
                            prob = (probability[last_position][index-1] +
                                    self.jointProbability_(
                                        sentence['words'],
                                        sentence['words'][last_position],
                                        sentence['words'][next_position],
                                        last_position+1, next_position))
                            if prob > probability[next_position][index]:
                                backtrace[next_position][index] = last_position
                                probability[next_position][index] = prob
                # Find the best sentence for each length
                for index in xrange(self.max_n_words):
                    end_marker = self.createWord_(self.markers['end'],
                                                  self.markers['end'])
                    best_score = -np.Infinity
                    best_position = -1
                    for position in xrange(len(sentence['words'])):
                        prob = (probability[position][index] +
                            self.jointProbability_(sentence['words'],
                                sentence['words'][position], end_marker,
                                position+1, len(sentence['words'])))
                        if prob > best_score:
                            best_score = prob
                            best_position = position
                    if best_score == -np.Infinity:
                        doc.ext['compressed_sentences'][-1].append((None,
                                                            best_score))
                    else:
                        doc.ext['compressed_sentences'][-1].append((
                            self.backtrace_(sentence['words'], backtrace,
                                            index, best_position), best_score))
        return documents

    def predict(self, documents):
        return None


class SentenceSelector(BaseEstimator):
    '''Selects the best compression of the first sentence inferior in length
    to max_length.'''
    def __init__(self, max_length=75):
        self.max_length = max_length

    def fit(self, documents, y=None):
        return self

    def predict(self, documents):
        output = []
        for doc in documents:
            q = [(-s[1], s[0]) for s in doc.ext['compressed_sentences'][0]]
            # stupid me...
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
