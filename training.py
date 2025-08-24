# ============================================================================
# SECTION 2: MODEL TRAINING AND EVALUATION
# ============================================================================

# 2.1 Import required libraries
import pandas as pd
import numpy as np
import os
import glob
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 2.2 Load the latest processed data
print("Loading processed data...")
train_files = glob.glob('processed_data/X_train_*.csv')
test_files = glob.glob('processed_data/X_test_*.csv')
le_files = glob.glob('processed_data/label_encoder_*.pkl')

if not train_files or not test_files or not le_files:
    print("Error: Processed data files not found. Please run DataPreprocessing.py first.")
    exit()

# Get the latest files based on timestamp
latest_train = max(train_files, key=os.path.getctime)
latest_test = max(test_files, key=os.path.getctime)
latest_le = max(le_files, key=os.path.getctime)

print(f"Loading training data: {latest_train}")
print(f"Loading test data: {latest_test}")
print(f"Loading label encoder: {latest_le}")

# Load the data
train_data = pd.read_csv(latest_train)
test_data = pd.read_csv(latest_test)

# Load label encoder
with open(latest_le, 'rb') as f:
    le = pickle.load(f)

# Separate features and target
X_train = train_data.drop('y_multi', axis=1)
y_train = train_data['y_multi']
X_test = test_data.drop('y_multi', axis=1)
y_test = test_data['y_multi']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Number of classes: {len(le.classes_)}")
print("Class names:", le.classes_)

# 2.3 Calculate class weights for imbalanced data
print(f"Training set size: {X_train.shape[0]} samples")
print("Calculating class weights for imbalanced data...")

# Calculate class weights using sklearn's balanced approach
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Create class weight dictionary
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Display class weights
print("Class weights:")
for class_idx, weight in class_weight_dict.items():
    class_name = le.inverse_transform([class_idx])[0]
    print(f"{class_name:<15}: {weight:.4f}")

# Apply scaling
print("\nScaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest classifier with class weights
print("Training RandomForest classifier with class weights...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,                # Keep reasonable number of trees
    max_depth=15,                    # Reasonable depth
    min_samples_split=10,            # Prevent overfitting
    min_samples_leaf=5,              # Prevent overfitting
    class_weight=class_weight_dict,  # Use calculated class weights
    random_state=42,
    n_jobs=-1,                       # Use all available cores
    verbose=1                        # Show progress
)

# 2.4 Train the model
print("\nTraining the multiclass model with class weights...")
rf_classifier.fit(X_train_scaled, y_train)

# 2.5 Make predictions on test set
print("Making predictions...")
y_pred = rf_classifier.predict(X_test_scaled)

# 2.6 Calculate and print metrics
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 2.7 Plot confusion matrix
print("\nPlotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
plt.figure(figsize=(10, 8))
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 2.8 Show class distribution and weights effectiveness
print(f"\nTraining set size: {X_train.shape[0]} samples")

# Get class distribution
class_dist = pd.Series(y_train).value_counts().sort_index()
class_dist.index = [le.inverse_transform([i])[0] for i in class_dist.index]
print("\nOriginal class distribution (imbalanced):")
print(class_dist.to_string())

print("\nClass weights applied to handle imbalance:")
for class_idx, weight in class_weight_dict.items():
    class_name = le.inverse_transform([class_idx])[0]
    count = class_dist[class_name]
    print(f"{class_name:<15}: {count:>8} samples, weight: {weight:.4f}")

# 2.9 Save results
# Create the directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save the predictions and true labels with class names
results = pd.DataFrame({
    'true_label': y_test,
    'true_label_name': le.inverse_transform(y_test),
    'predicted_label': y_pred,
    'predicted_label_name': le.inverse_transform(y_pred),
    'correct': (y_test == y_pred)
})

# Save results to CSV
results_file = f'processed_data/multiclass_predictions_{timestamp}.csv'
results.to_csv(results_file, index=False)

# Save class mapping
class_mapping = pd.DataFrame({
    'class_index': range(len(le.classes_)),
    'class_name': le.classes_
})
class_mapping.to_csv('processed_data/class_mapping.csv', index=False)

print(f"\nMulticlass predictions saved to: {os.path.abspath(results_file)}")
print(f"Class mapping saved to: {os.path.abspath('processed_data/class_mapping.csv')}")

print("\n" + "="*80)
print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
print("="*80)
