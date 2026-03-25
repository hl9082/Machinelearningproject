"""
DTEncoding.py

Author: Jesse Dinh


"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

workingLanguages = ["English", "French", "Spanish"]

function_words = [
    # English
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "on", "with",
    # Spanish
    "el", "la", "las", "un", "una", "para", "en", "y", "que", "con", "de", "es",
    # French
    "le", "les", "et", "pour", "de", "du", "des", "est", "une", "en"
]

texts = []
labels = []

for language in workingLanguages:
    with open(language + ".csv", "r") as file:
        next(file)
        for line in file:
            line = line.strip().lower()
            if line:
                texts.append(line)
                labels.append(language)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
vectorizer = CountVectorizer(vocabulary=function_words)

x = vectorizer.fit_transform(texts).toarray()

np.save("X.npy", x)
np.save("Y.npy", y)
