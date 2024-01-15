import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_data=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train_data.head(20)

train_data.info(),train_data.shape

train_data.describe(include = 'object')

# Drop rows with NaN values in the 'keyword' column
train_data.dropna(subset=['keyword'], inplace=True)

# Find the mode of the 'keyword' column
keyword_mode = train_data['keyword'].mode()

# Check if there is a mode other than the first one
if len(keyword_mode) > 1:
    # Fill NaN values in 'keyword' with the second most common value
    train_data['keyword'].fillna(keyword_mode[1], inplace=True)
train_data["keyword"].isnull().sum()

train_data.head()

# Handling missing values
train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# Text cleaning (you can add more advanced cleaning techniques)
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\n', ' ', text)     # Remove newline characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

train_data['clean_text'] = train_data['text'].apply(clean_text)
test_data['clean_text'] = test_data['text'].apply(clean_text)

# Feature selection
features = ['clean_text', 'keyword', 'location']  # Define features for model training
X_train = train_data[features]
y_train = train_data['target']
X_test = test_data[features]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model, you can use other models

# Create a pipeline for text vectorization and model training
text_clf = make_pipeline(
    TfidfVectorizer(max_features=9000),  # Adjust max_features as needed
    RandomForestClassifier(random_state=47)  # You can use other classifiers
)

# Split the training data for model evaluation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=47)

# Fit the model on the training data
text_clf.fit(X_train_split['clean_text'], y_train_split)

# Predict on the validation set to evaluate the model
val_predictions = text_clf.predict(X_val['clean_text'])

# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# Retrain on the full training data
text_clf.fit(X_train['clean_text'], y_train)

# Predict on test data
test_predictions = text_clf.predict(X_test['clean_text'])

# Create submission file in the required format (assuming 'id' column in test_data)
submission_df = pd.DataFrame({'id': test_data['id'], 'target': test_predictions})
submission_df.to_csv('submission.csv', index=False)