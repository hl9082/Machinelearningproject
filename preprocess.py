import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import regex as rg

SAMPLE_LENGTH = 15

# match strings with only latin characters
# accents, digits, symbols, and punctuation are allowed
validChars = rg.compile(r'[\p{IsLatin}\s\p{P}\d]')

def break_down(text, length):
    """Split long strings into shorter strings with uniform length.

    :param text: Any string.
    :type text: str
    :param length: Desired length of each substring.
    :type length: int
    :return: A list of substrings.
    :rtype: str[]
    """
    output = []
    words = text.split()
    i = 0
    buffer = ""
    for word in words:
        buffer = buffer + word
        i+=1
        if i >= length:
            output.append(buffer)
            buffer = ""
            i = 0
        else:
            buffer += " "
    return output



raw = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 
                              "zarajamshaid/language-identification-datasst",
                              "dataset.csv")

# more languages will be added later
workingLanguages = ["English", "French", "Spanish"]
langData = []
for i in workingLanguages:
    langData.append([])


for row in raw.itertuples():
    for i in range(len(workingLanguages)):
        if row.language == workingLanguages[i]:
            langData[i].extend(break_down(row.Text, SAMPLE_LENGTH))
            break

for i in range(len(workingLanguages)):
    with open(workingLanguages[i] + ".csv", "w") as file:
        file.write(workingLanguages[i] + "\n")
        for line in langData[i]:
            if validChars.match(line):
                file.write(line + "\n")

