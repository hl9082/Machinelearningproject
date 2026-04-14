"""
predict.py

Author: Huy Le (hl9082)

Inference pipeline for the decision tree classifier.
Loads the saved .joblib model and uses the exact same CountVectorizer 
vocabulary and LabelEncoder mapping from DTEncoding.py to predict
the language of unseen text.
"""

import joblib # For loading the saved model
from sklearn.feature_extraction.text import CountVectorizer # Convert a collection of text documents to a matrix of token counts
from pathlib import Path # For handling file paths

class ModelComparator:
    """
    Loads multiple trained models to compare their live inferences.
    Attributes:
        function_words (list): The fixed vocabulary of function words used for vectorization.
        vectorizer (CountVectorizer): The CountVectorizer instance initialized with the function words.
        label_map (dict): A mapping from integer labels to language names.
        models (dict): A dictionary of loaded machine learning models for inference.
    """

    def __init__(self, models_dir):
        self.function_words = [
            "the", "and", "of", "to", "in", "is", "that", "it", "for", "on", "with",
            "el", "la", "las", "un", "una", "para",  "y", "que", "con", "es",
            "le", "les", "et", "pour", "de", "du", "des", "est", "une", "en"
        ] # Must match the vocabulary used during training in DTEncoding.py
        self.vectorizer = CountVectorizer(vocabulary=self.function_words) # Must be initialized with the same fixed vocabulary as during training
        self.label_map = {0: "English", 1: "French", 2: "Spanish"}
        
        # Load all three models
        self.models = {
            "Decision Tree": joblib.load(models_dir / "Decision_Tree.joblib"),
            "Random Forest": joblib.load(models_dir / "Random_Forest.joblib"),
            "Neural Network": joblib.load(models_dir / "Neural_Network.joblib")
        }

    def predict_all(self, text):
        """
        Predicts the language of the input text using all loaded models and prints the results.
        Args:
            text (str): The input text to be classified.
        Returns:
            None: Prints the predicted language for each model to the console.
        """
        if not text.strip():
            return "Error: Empty text."

        '''
        This statement below transforms documents to document-term matrix.
        It extracts token counts out of raw text documents using the vocabulary 
        fitted with fit or the one provided to the constructor.
        '''
        vectorized_text = self.vectorizer.transform([text]).toarray()
        
        
        print(f"Text: '{text}'")
        # Ask each model for its prediction
        for name, model in self.models.items():
            pred_int = model.predict(vectorized_text)[0]
            pred_lang = self.label_map.get(pred_int, "Unknown")
            print(f"  > {name:15}: {pred_lang}")
        print("-" * 50)

if __name__ == "__main__":
    dir_path = Path(__file__).resolve().parent
    
    try:
        comparator = ModelComparator(models_dir=dir_path)
        
        # this is the array of strings from the 3 different languages to test the models
        test_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "El zorro marrón rápido salta sobre el perro perezoso.",
            "Le renard brun rapide saute par-dessus le chien paresseux.",
            "The le un and pour" # A tricky edge case to see if they disagree!
        ]
        
        for sample in test_samples:
            comparator.predict_all(sample)
            
    except FileNotFoundError as e:
        print(f"Error loading models. Have you run the updated train_models.py? Details: {e}")