"""Preprocesses a language identification dataset from Kaggle.

This script downloads a dataset containing text in various languages, bypasses 
standard Pandas CSV parsing to avoid buffer overflows from malformed lines, 
and extracts text into uniform word-length chunks. It then filters out any 
chunks with invalid characters and exports the cleaned data into individual 
CSV files per language within a local 'data' directory.

Author:
    Noah Shin
Co-author: Huy Le
"""

import os
import pandas as pd
import kagglehub
import regex as rg

# The number of words desired per extracted line of text
SAMPLE_LENGTH = 15

# Matches strings with only Latin characters, spaces, punctuation, and digits
validChars = rg.compile(r'[\p{IsLatin}\s\p{P}\d]+$')


def break_down(text, length):
    """Splits long strings into shorter strings of a uniform word length.

    Args:
        text (str): The input text to be split.
        length (int): The desired number of words in each returned substring.

    Returns:
        list: A list of string chunks, where each chunk contains exactly 
            `length` words (remaining words that do not meet the exact 
            length requirement are dropped).
    """
    output = []
    words = str(text).split()
    i = 0
    buffer = ""
    for word in words:
        buffer = buffer + word
        i += 1
        if i >= length:
            output.append(buffer)
            buffer = ""
            i = 0
        else:
            buffer += " "
    return output


def main():
    """Executes the main data loading, processing, and exporting pipeline.
    
    Downloads the raw dataset, manually parses the CSV to handle malformed 
    quotes, loads the valid rows into a Pandas DataFrame, and processes 
    specifically targeted languages into independent output files inside 
    a 'data/' folder.
    """
    # --- 1. Bulletproof Dataset Loading ---
    # Download the dataset directly (returns the folder path)
    dataset_path = kagglehub.dataset_download("zarajamshaid/language-identification-datasst")
    csv_path = f"{dataset_path}/dataset.csv"

    # Read the file manually to completely bypass Pandas' strict CSV rules
    records = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f, None)  # Skip the header row
        for line in f:
            # Split line by commas, ignoring quote rules entirely
            row = line.strip().split(',')
            if len(row) >= 2:
                # The language label is always the very last item
                lang = row[-1].strip().strip('"\'').capitalize()
                # The text is everything else, glued back together
                text = ",".join(row[:-1]).strip().strip('"\'')
                records.append({'Text': text, 'language': lang})

    # Now load the cleanly parsed data into Pandas
    raw = pd.DataFrame(records)

    # --- 2. Process & Export ---
    workingLanguages = ["English", "French", "Spanish"]

    # --- Directory Management ---
    # Find the folder containing this script (the 'code/' folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the main project directory
    project_root = os.path.dirname(script_dir)

    # Define the target 'data/' folder path
    data_dir = os.path.join(project_root, "data")
    
    # Safely create the 'data/' directory if it doesn't already exist
    os.makedirs(data_dir, exist_ok=True)

    for lang in workingLanguages:
        # 1. Filter by language
        df_lang = raw[raw['language'] == lang]
        
        if df_lang.empty:
            print(f"Warning: No data found for '{lang}'.")
            continue
            
        # 2. Break down into 15-word chunks using your function
        chunks = df_lang['Text'].apply(lambda x: break_down(x, SAMPLE_LENGTH)).explode()
        chunks = chunks.dropna()
        
        # 3. Filter using your validChars regex
        valid_chunks = chunks[chunks.apply(lambda x: bool(validChars.fullmatch(str(x))))]
        
        # 4. Save to CSV inside the data/ folder
        if valid_chunks.empty:
            print(f"Warning: Data was found for '{lang}', but none passed the 15-word/regex filter.")
        else:
            # Safely join the folder and filename (e.g., 'data/English.csv')
            out_path = os.path.join(data_dir, f"{lang}.csv")
            valid_chunks.to_csv(out_path, index=False, header=[lang], encoding="utf-8")
            print(f"Successfully saved {len(valid_chunks)} valid 15-word lines to {out_path}")


if __name__ == "__main__":
    main()