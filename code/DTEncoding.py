"""
DTEncoding.py

Author: Jesse Dinh


"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

workingLanguages = ["English", "French", "Spanish", "Dutch", "Portugese", "Estonian"]

function_words = [
    # English
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "on", "with",
    # Spanish
    "el", "la", "las", "un", "una", "para",  "y", "que", "con", "es",
    # French
    "le", "les", "et", "pour", "de", "du", "des", "est", "une", "en",
    # Portuguese
    "a", "o", "e", "do", "da", "em", "os", "as", "no", "na",
    # Dutch
    "van", "het", "een", "op", "met", "voor", "zijn", "te", "door", "werd",
    # Estonian
    "ja", "ei", "see", "ta", "kui", "ka", "oli", "või", "oma", "mis"
]

texts = []
labels = []

for language in workingLanguages:
    with open("../data/" + language + ".csv", "r", encoding ="utf-8") as file:
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
