CSCI 335 Group Project

Title: Language Classification Model

Group: Jesse Dinh, Huy Le, Noah Shin

## Abstract: 
This project builds a machine learning model that can classify inputted text data into one of multiple languages. Our project aims to use supervised learning to train a model on labeled training data and is tested on unseen new text data. We preprocess the data and transform it into numerical feature vectors with the top function words in each language.

## Project Structure
The pipeline is broken down into four distinct stages:
1. **Data Preprocessing** (`preprocess.py`): Downloads the raw dataset via KaggleHub, filters for valid characters, and normalizes text into 15-word chunks.
2. **Feature Encoding** (`DTEncoding.py`): Reads the processed data and utilizes a `CountVectorizer` to map text to a predefined vocabulary of function words, exporting the results as NumPy arrays (`X.npy` and `Y.npy`).
3. **Model Training** (`TrainDT.py`): Splits the encoded arrays into training and testing sets, trains a standard CART Decision Tree, evaluates its accuracy, and serializes the model into a `.joblib` file.
4. **Inference Pipeline** (`predict.py`): Loads the serialized model and processes new, unseen user text to predict its language.

## How to Run

**1. Clone the repository and install dependencies**
```bash
git clone https://github.com/hl9082/Machinelearningproject.git
cd Machinelearningproject
python -m venv <name>
<name>\scripts\Activate

pip install -r requirements.txt
(Ensure your requirements.txt includes scikit-learn, numpy, pandas, kagglehub, regex, and joblib)

**2. Fetch and Preprocess the Data**
Run the preprocessing script to download the dataset and generate the language-specific CSV files.
python preprocess.py
(Note: Ensure the output CSVs are moved to the ../data/ directory, or update the paths in DTEncoding.py to match your directory structure.)

**3. Encode the Features**
Convert the CSV text data into numerical vectors. This will generate X.npy and Y.npy in your working directory.

```
python DTEncoding.py
```

**4. Train the Model**
Train the decision tree classifier. This script will print the model's accuracy on the test split and save the trained model as DT.joblib.

```
python TrainDT.py
```

**5. Test the Model (Inference)**
Run the prediction script to test the model on new, unseen text snippets.
```
python predict.py
```

