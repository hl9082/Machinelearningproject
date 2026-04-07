"""
TrainDT.py

Authors: Noah Shin, Huy Le

Given a numpy array of labels loaded from Y.npy and a numpy
array of encoded vectors loaded from X.npy, trains a decision
tree classifier and saves it as a .joblib file.

This script trains, tunes, and evaluates three distinct machine learning models:
1. Decision Tree (Traditional)
2. Random Forest (Traditional / Ensemble)
3. Multi-Layer Perceptron (Neural Network)

It uses GridSearchCV for hyperparameter tuning and cross-validation, 
evaluates each model on a test set, and saves the best performing model 
as a .joblib file for production inference.

"""

import sklearn as sk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from pathlib import Path
import joblib
import warnings

# Ignore convergence warnings from MLPClassifier for cleaner terminal output
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

'''
dir = Path(__file__).resolve().parent

vectors = np.load(dir/"X.npy")
labels = np.load(dir/"Y.npy")

v_train, v_test, l_train, l_test = train_test_split(vectors, labels, test_size=0.2)

dt = DecisionTreeClassifier(max_depth=10)
dt.fit(v_train, l_train)
l_predicted = dt.predict(v_test)
print("Accuracy: " + str(accuracy_score(l_test, l_predicted)))

joblib.dump(dt, dir/"DT.joblib")
'''

def main():
    """Main execution function for the training and evaluation pipeline."""
    
    # 1. Setup and Data Loading
    dir_path = Path(__file__).resolve().parent
    try:
        vectors = np.load(dir_path / "X.npy")
        labels = np.load(dir_path / "Y.npy")
    except FileNotFoundError:
        print("Error: Could not find X.npy or Y.npy. Please run DTEncoding.py first.")
        return

    # Random state added for reproducibility
    v_train, v_test, l_train, l_test = train_test_split(
        vectors, labels, test_size=0.2, random_state=42
    )
    print(f"Data successfully loaded and split. Training set size: {len(v_train)} samples.\n")

    # 2. Define the Models and their Hyperparameter Grids
    # We define a dictionary where the key is the model name, and the value is 
    # another dictionary containing the base estimator and the grid of parameters to test.
    models_config = {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        },
        "Neural Network (MLP)": {
            # max_iter is increased slightly to give the network time to converge
            "estimator": MLPClassifier(max_iter=500, random_state=42),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "learning_rate_init": [0.001, 0.01]
            }
        }
    }

    # 3. Training, Tuning, and Evaluation Loop
    best_overall_model = None
    best_overall_accuracy = 0.0
    best_overall_name = ""

    print("=== Starting Model Training & Hyperparameter Tuning ===\n")

    for model_name, config in models_config.items():
        print(f"Training {model_name}...")
        
        # Initialize GridSearchCV (cv=5 means 5-fold cross-validation)
        # n_jobs=-1 utilizes all available CPU cores for faster training
        grid_search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit all combinations of hyperparameters
        grid_search.fit(v_train, l_train)
        
        # Extract the best model from the grid search
        best_model_for_type = grid_search.best_estimator_
        
        # Evaluate on the unseen test set
        l_predicted = best_model_for_type.predict(v_test)
        test_accuracy = accuracy_score(l_test, l_predicted)
        
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Optional: Print detailed classification report to see precision/recall per language
        print(classification_report(l_test, l_predicted))
        print("-" * 50)

        # ADD THIS: Save each tuned model individually
        file_name = model_name.replace(" ", "_").replace("_(MLP)", "") + ".joblib"
        joblib.dump(best_model_for_type, dir_path / file_name)
        
        # Check if this is the highest scoring model we've seen so far
        if test_accuracy > best_overall_accuracy:
            best_overall_accuracy = test_accuracy
            best_overall_model = best_model_for_type
            best_overall_name = model_name

    # 4. Model Selection and Serialization
    print("=== Training Complete ===")
    print(f"Champion Model: {best_overall_name} with Accuracy of {best_overall_accuracy:.4f}")
    
    # Save only the winning model
    output_path = dir_path / "best_model.joblib"
    joblib.dump(best_overall_model, output_path)
    print(f"\nWinning model saved to: {output_path}")

if __name__ == "__main__":
    main()