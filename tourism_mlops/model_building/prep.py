# for data manipulation
import pandas as pd
import numpy as np
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi
from pathlib import Path

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "data" / "tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Data cleaning and preprocessing
print("\n=== Data Cleaning ===")

# Remove unnecessary columns
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)
    print("Removed CustomerID column")

# Clean Gender column inconsistencies
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print("Fixed Gender column inconsistencies")

# Handle missing values
print(f"Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Fill missing values with appropriate strategies
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object']).columns

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled")

# Encode categorical variables
label_encoders = {}
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
    'MaritalStatus', 'Designation'
]

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}")

# Define target variable
target_col = 'ProdTaken'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeature columns: {list(X.columns)}")
print(f"Target distribution: \n{y.value_counts()}")

# Perform stratified train-test split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Train target distribution: \n{y_train.value_counts()}")
print(f"Test target distribution: \n{y_test.value_counts()}")

# Save datasets locally
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nDatasets saved locally")

# Upload files to Hugging Face Hub (optional)
if os.getenv("HF_TOKEN"):
    try:
        files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

        for file_path in files:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,  # filename in the repo
                repo_id="shashidj/tourism-package-prediction",
                repo_type="dataset",
            )
            print(f"Uploaded {file_path} to Hugging Face Hub")
    except Exception as e:
        print(f"Warning: Could not upload to HuggingFace Hub: {e}")
        print("Note: HF_TOKEN may not be set or repository may not exist")
        print("You can still proceed with local files for model training")
else:
    print("Note: HF_TOKEN not set. Skipping upload to HuggingFace Hub")
    print("You can still proceed with local files for model training")

print("\n=== Data Preparation Complete ===")
print(f"Processed {len(df)} records")
print(f"Features: {len(X.columns)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
