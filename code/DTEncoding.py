"""
DTEncoding.py

Convert the CSV text data into numerical vectors. This will generate X.npy and Y.npy in your working directory.

Author: Jesse Dinh

Co-authors: Huy Le, Noah Shin

"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

workingLanguages = ["English", "French", "Spanish"] # The order of this list must match the alphabetical order of the languages as they are encoded by LabelEncoder in predict.py (English=0, French=1, Spanish=2)

# Define the fixed vocabulary of function words used for vectorization. This must match the list used during training in predict.py.
function_words = [
    # English
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "on", "with",
    # Spanish
    "el", "la", "las", "un", "una", "para",  "y", "que", "con", "es",
    # French
    "le", "les", "et", "pour", "de", "du", "des", "est", "une", "en"
]

texts = [] # List to hold all text documents from the CSV files
labels = [] # List to hold the corresponding language labels for each text document

# Read the cleaned CSV files for each language and prepare the text and label lists
for language in workingLanguages:
    with open("../data/" + language + ".csv", "r", encoding ="utf-8") as file:
        next(file)
        for line in file:
            line = line.strip().lower()
            if line:
                texts.append(line)
                labels.append(language)
label_encoder = LabelEncoder() # Encode the string labels into integers (e.g., English=0, French=1, Spanish=2)
y = label_encoder.fit_transform(labels) # Convert the list of text documents into a matrix of token counts using the fixed vocabulary of function words
vectorizer = CountVectorizer(vocabulary=function_words) # Must be initialized with the same fixed vocabulary as during training

x = vectorizer.fit_transform(texts).toarray() # Convert the sparse matrix to a dense array and save the feature vectors and labels as .npy files for later use in model training.

# Save the feature vectors and labels as .npy files for later use in model training.
np.save("X.npy", x)
np.save("Y.npy", y)
