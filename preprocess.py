import nltk
import re
import os
import pandas as pd
from collections import defaultdict
from num2words import num2words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Preprocessing all documents with listed 6 steps
def preprocess(input_text):
    # 1. Convert to lowercase
    lowercase_input = input_text.lower()

    # 4. Single Characters
    input_without_char = re.sub(r'\s+[a-zA-Z]\s+', ' ', lowercase_input)

    # 2. Remove stop words
    text_tokens = word_tokenize(input_without_char)
    input_without_sw = [word for word in text_tokens if word not in stopwords.words()]

    # 3. Remove Punctuation
    input_without_punc = [word for word in input_without_sw if word.isalnum()]

    # 5. Stemming and Lemmatization, e.g., change “playing” and “played” to play
    # WordNetLemmatizer needs Pos tags to understand if word is noun, verb or adj etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    final_words = []
    word_lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(input_without_punc):
        word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
        final_words.append(word_final)

    # 6. Converting Numbers, e.g., “1000” to one thousand
    for i in range(0, len(final_words)):
        if final_words[i].isdigit():
            final_words[i] = num2words(final_words[i])

    return final_words

if __name__ == '__main__':
    if os.path.exists("preprocessed.csv"):
        df = pd.read_csv("preprocessed.csv")
    else:
        df = pd.read_csv("deceptive-opinion.csv")
        df["preprocessed_text"] = 'default value'

        for ind in df.index:
            return_value = preprocess(df['text'][ind])
            df['preprocessed_text'][ind] = ' '.join(return_value)

        df.to_csv("preprocessed.csv", index=False)