import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Find the latest processed data file
list_of_files = glob.glob('processed_data/*.csv')
if not list_of_files:
    print("No processed data file found. Please run BinaryDataPreprocessing.py first.")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"Loading latest processed data: {latest_file}")
data = pd.read_csv(latest_file)

# Prepare features (X) and target (y) for binary classification
X = data.drop('Attack Type', axis=1)
y = (data['Attack Type'] != 'BENIGN').astype(int) # 1 for intrusion, 0 for benign

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Apply Incremental PCA on the training set
print("Applying PCA on the training set...")
n_components = X_train.shape[1] // 2
ipca = IncrementalPCA(n_components=n_components, batch_size=1000)

# Fit PCA on the training data and transform it
X_train_pca = ipca.fit_transform(X_train)

# Transform the test data using the fitted PCA
X_test_pca = ipca.transform(X_test)

print(f'Information retained after PCA: {sum(ipca.explained_variance_ratio_):.2%}')
print(f"Shape after PCA: {X_train_pca.shape}")

# Train a RandomForestClassifier
print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_pca, y_train)

# Evaluate the model
print("\nEvaluating model on the test set...")
y_pred = clf.predict(X_test_pca)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'Intrusion']))