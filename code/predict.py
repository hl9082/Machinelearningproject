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

class ModelComparator:
    """Loads multiple trained models to compare their live inferences."""

    def __init__(self, models_dir):
        self.function_words = [
            "the", "and", "of", "to", "in", "is", "that", "it", "for", "on", "with",
            "el", "la", "las", "un", "una", "para",  "y", "que", "con", "es",
            "le", "les", "et", "pour", "de", "du", "des", "est", "une", "en"
        ]
        self.vectorizer = CountVectorizer(vocabulary=self.function_words)
        self.label_map = {0: "English", 1: "French", 2: "Spanish"}
        
        # Load all three models
        self.models = {
            "Decision Tree": joblib.load(models_dir / "Decision_Tree.joblib"),
            "Random Forest": joblib.load(models_dir / "Random_Forest.joblib"),
            "Neural Network": joblib.load(models_dir / "Neural_Network.joblib")
        }

    def predict_all(self, text):
        if not text.strip():
            return "Error: Empty text."

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