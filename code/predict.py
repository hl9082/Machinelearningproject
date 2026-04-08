"""
predict.py

Author: Huy Le (hl9082)

Inference pipeline for the decision tree classifier.
Loads the saved .joblib model and uses the exact same CountVectorizer 
vocabulary and LabelEncoder mapping from DTEncoding.py to predict
the language of unseen text.
"""

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

class LanguageClassifier:
    """Inference pipeline for the language decision tree classifier.

    This class loads a pre-trained decision tree model and reconstructs the 
    exact text preprocessing pipeline (vocabulary and label encoding) used 
    during the training phase to ensure accurate predictions on new data.

    Attributes:
        model (sklearn.tree.DecisionTreeClassifier): The loaded decision tree model 
            used for inference.
        function_words (list of str): The hardcoded list of function words used as 
            features, maintaining the exact order from training.
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): The vectorizer 
            used to transform raw strings into numerical feature arrays.
        label_map (dict): A mapping from integer model predictions back to 
            human-readable language names.
    """

    def __init__(self, model_path):
        """Initializes the LanguageClassifier.

        Loads the trained model from disk and recreates the vocabulary and 
        label mappings used by DTEncoding.py during the data preparation phase.

        Args:
            model_path (str or pathlib.Path): The absolute or relative file path 
                pointing to the trained DT.joblib file.

        Raises:
            FileNotFoundError: If the model file cannot be found at the specified path.
        """
        # 1. Load the trained decision tree model
        self.model = joblib.load(model_path)
        
        # 2. Recreate the exact vocabulary from DTEncoding.py
        self.function_words = [
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
        
        # Initialize CountVectorizer exactly as Jesse did
        self.vectorizer = CountVectorizer(vocabulary=self.function_words)
        
        # 3. Recreate the LabelEncoder mapping
        # sklearn's LabelEncoder sorts alphabetically: English=0, French=1, Spanish=2
        self.label_map = {
            0: "English",
            1: "French",
            2: "Spanish",
            3:  "Dutch",
            4:  "Portugese",
            5:  "Estonian"
        }

    def predict(self, text):
        """Predicts the language of a given text string.

        Converts the raw input string into a numerical feature vector using the 
        CountVectorizer, passes it to the loaded decision tree model, and maps 
        the resulting integer prediction to a language name.

        Args:
            text (str): The raw text string to be classified.

        Returns:
            str: The predicted language (e.g., "English", "French", "Spanish"). 
                 Returns an error string if the input text is empty or purely whitespace.
        """
        if not text.strip():
            return "Error: Empty text provided."

        # Step 1: Vectorize the text 
        # (CountVectorizer expects an iterable of strings, hence the list bracket)
        vectorized_text = self.vectorizer.transform([text]).toarray()
        
        # Step 2: Pass the 2D feature array to the model
        prediction_int = self.model.predict(vectorized_text)[0]
        
        # Step 3: Map the integer back to a human-readable language string
        predicted_language = self.label_map.get(prediction_int, "Unknown Language")
        
        return predicted_language

# ==========================================
# Execution / Testing
# ==========================================
if __name__ == "__main__":
    # Resolve path dynamically to handle relative execution
    dir_path = Path(__file__).resolve().parent
    model_file = dir_path / "DT.joblib"
    
    try:
        pipeline = LanguageClassifier(model_path=model_file)
        
        # Test with new, unseen text snippets
        test_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "El zorro marrón rápido salta sobre el perro perezoso.",
            "Le renard brun rapide saute par-dessus le chien paresseux."
        ]
        
        for sample in test_samples:
            print(f"Text: '{sample}'")
            print(f"Prediction: {pipeline.predict(sample)}")
            print("-" * 50)
            
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_file}. Did you run TrainDT.py first?")