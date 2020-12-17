from .feature_extraction import calculate_pos_freq
from sklearn.feature_extraction.text import CountVectorizer

def predict(model, review, feature, vectorizer=None):
    if feature == 'bag-of-words':
        new_cv = CountVectorizer(vocabulary=vectorizer.get_feature_names(), max_features=1000, ngram_range=(1, 2))
        vector = new_cv.fit_transform([review]).toarray()
    elif feature == 'tf-idf':
        vector = vectorizer.fit_transform([review])
        vector = vector.todense()
    elif feature == 'pos-tag-freq':
        vector = calculate_pos_freq(data=[review])

    y_review = model.predict(vector)
    return y_review
