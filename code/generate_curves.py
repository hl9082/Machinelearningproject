'''
Author: Huy Le
Description: This script generates learning curves for the Decision Tree and Random Forest models, and a loss curve for the Multi-Layer Perceptron (MLP) model. The curves are saved as high-resolution PNG files in the same directory as the script.
'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import json

def plot_learning_curve(estimator, title, X, y, output_dir):
    """
    Generates a learning curve (Accuracy vs Training Size) for traditional models.
    Args:
        estimator: The machine learning model (e.g., DecisionTreeClassifier or RandomForestClassifier).
        title: The title for the plot (used for the title and filename).
        X: The feature matrix.
        y: The target labels.
        output_dir: The directory where the PNG file will be saved.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)
    plt.xlabel("Training Examples", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
    plt.legend(loc="best")
    
    safe_title = title.replace(" ", "_")
    plt.savefig(output_dir / f"{safe_title}.png", dpi=300)
    plt.close()

def plot_mlp_loss_curve(estimator, X, y, output_dir):
    """
    Trains an MLP and extracts its loss curve (Loss vs Epochs).
    
    Args:
        estimator: the ML model used here, which is MLP.
        X: The feature matrix.
        y: The target labels.
        output_dir: The directory where the PNG file will be saved.
    
    """
    # Using the best params we found earlier for a clean chart
    estimator.fit(X, y) 
    
    plt.figure(figsize=(8, 6))
    plt.title("Neural Network Loss Curve (Over Epochs)", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Log-Loss)", fontsize=12)
    plt.grid(True)

    plt.plot(estimator.loss_curve_, color="blue", linewidth=2, label="Training Loss")
    plt.legend(loc="best")
    
    plt.savefig(output_dir / "Neural_Network_Loss_Curve.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    dir_path = Path(__file__).resolve().parent
    try:
        X = np.load(dir_path / "X.npy")
        y = np.load(dir_path / "Y.npy")
    except FileNotFoundError:
        print("Data files not found. Please run DTEncoding.py first.")
        exit()

    # Load the master parameters file
    params_file = dir_path / "all_best_params.json"
    try:
        with open(params_file, "r") as f:
            all_params = json.load(f)
            
            # JSON converts tuples to lists. We must convert the MLP hidden layers back to a tuple.
            if "hidden_layer_sizes" in all_params.get("Neural Network (MLP)", {}):
                all_params["Neural Network (MLP)"]["hidden_layer_sizes"] = tuple(
                    all_params["Neural Network (MLP)"]["hidden_layer_sizes"]
                )
    except FileNotFoundError:
        print("Error: all_best_params.json not found. Run train_models.py first.")
        exit()

    print("Generating Decision Tree Learning Curve...")
    dt_params = all_params.get("Decision Tree", {})
    plot_learning_curve(
        DecisionTreeClassifier(**dt_params, random_state=42), 
        "Decision Tree Learning Curve", X, y, dir_path
    )

    print("Generating Random Forest Learning Curve...")
    rf_params = all_params.get("Random Forest", {})
    plot_learning_curve(
        RandomForestClassifier(**rf_params, random_state=42), 
        "Random Forest Learning Curve", X, y, dir_path
    )

    print("Generating Neural Network Loss Curve...")
    mlp_params = all_params.get("Neural Network (MLP)", {})
    plot_mlp_loss_curve(
        MLPClassifier(**mlp_params, max_iter=500, random_state=42), 
        X, y, dir_path
    )
    
    print("All curves generated and saved successfully!")