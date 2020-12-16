import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

def calculate_bag_of_words(data, max_features=1000, ngram=1):
    '''
        input - data: a set of reviews,
                max_features: no. of words to be selected
                ngram: specifies how many words can be selected together as a feature
        :return: matrix(rows are reviews, cols are feature words
                 M[3][2]: no. of times word_2 appears in review_3
                 Hint: output matrix can directly be split into training and testing data
    '''

    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, ngram))
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors

# Extract TF-IDF features for all documents
def calculate_tf_idf(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    tfidf = pd.DataFrame(denselist, columns=feature_names)
    return tfidf

def calculate_pos_freq(data):
    '''
        input - original text

        :return:
    '''

    tagset = {
        'CC': 'conjunction, coordinating',
        'CD': 'numeral, cardinal',
        'DT': 'determiner',
        'EX': 'existential there',
        'IN': 'preposition or conjunction, subordinating',
        'JJ': 'adjective or numeral, ordinal',
        'JJR': 'adjective, comparative',
        'JJS': 'adjective, superlative',
        'LS': 'list item marker',
        'MD': 'modal auxiliary',
        'NN': 'noun, common, singular or mass',
        'NNP': 'noun, proper, singular',
        'NNS': 'noun, common, plural',
        'PDT': 'pre - determiner',
        'POS': 'genitive marker',
        'PRP': 'pronoun, personal',
        'PRP$': 'pronoun, possessive',
        'RB': 'adverb',
        'RBR': 'adverb, comparative',
        'RBS': 'adverb, superlative',
        'RP': 'particle',
        'TO': '-to- as preposition or infinitive marker',
        'UH': 'interjection',
        'VB': 'verb, base form',
        'VBD': 'verb, past tense',
        'VBG': 'verb, present participle or gerund',
        'VBN': 'verb, past participle',
        'VBP': 'verb, present tense, not 3rd person singular',
        'VBZ': 'verb, present tense, 3rd person singular',
        'WDT': 'WH-determiner',
        'WP': 'WH-pronoun',
        'WRB': 'Wh-adverb'
    }
    arr = []
    for d in data:
        # Convert to lowercase
        lowercase_data = d.lower()
        tokens = word_tokenize(lowercase_data)
        tags = pos_tag(tokens)
        new_tags = [tag for tag in tags if tag[0] not in stopwords.words() and len(tag[0]) > 1]
        counts = Counter(tag for word, tag in new_tags)
        a = []
        for k in tagset.keys():
            a.append(counts[k])

        arr.append(a)

    return arr






