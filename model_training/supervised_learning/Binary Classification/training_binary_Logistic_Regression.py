
import pandas as pd
import numpy as np
import os
import glob
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create the directory if it doesn't exist
os.makedirs('results_binary/LogisticRegression', exist_ok=True)

# 2.2 Load the latest processed data
print("Loading processed data...")
train_files = glob.glob('model_training/data/processed_data_binary/X_train_scaled_binary_class_*.csv')
test_files = glob.glob('model_training/data/processed_data_binary/X_test_scaled_binary_class_*.csv')
le_files = glob.glob('model_training/data/processed_data_binary/label_encoder_binary_class_*.pkl')
protocol_encoder_files = glob.glob('model_training/data/processed_data_binary/protocol_encoder_binary_class_*.pkl')
state_encoder_files = glob.glob('model_training/data/processed_data_binary/state_encoder_binary_class_*.pkl')

if not train_files or not test_files or not le_files:
    print("Error: Processed data files not found. Please run DataPreprocessing.py first.")
    exit()

# Get the latest files based on timestamp
latest_train = max(train_files, key=os.path.getctime)
latest_test = max(test_files, key=os.path.getctime)
latest_le = max(le_files, key=os.path.getctime)

# Get latest protocol and state encoders if they exist
latest_protocol_encoder = None
latest_state_encoder = None
protocol_encoder = None
state_encoder = None

if protocol_encoder_files:
    latest_protocol_encoder = max(protocol_encoder_files, key=os.path.getctime)
    print(f"Loading protocol encoder: {latest_protocol_encoder}")

if state_encoder_files:
    latest_state_encoder = max(state_encoder_files, key=os.path.getctime)
    print(f"Loading state encoder: {latest_state_encoder}")

print(f"Loading training data: {latest_train}")
print(f"Loading test data: {latest_test}")
print(f"Loading label encoder: {latest_le}")

# Load the data
train_data = pd.read_csv(latest_train)
test_data = pd.read_csv(latest_test)

# Load label encoder
with open(latest_le, 'rb') as f:
    le = pickle.load(f)

# Load protocol encoder if available
if latest_protocol_encoder:
    with open(latest_protocol_encoder, 'rb') as f:
        protocol_encoder = pickle.load(f)
    print("Protocol encoder loaded successfully")

# Load state encoder if available
if latest_state_encoder:
    with open(latest_state_encoder, 'rb') as f:
        state_encoder = pickle.load(f)
    print("State encoder loaded successfully")



# Separate features and target
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# 2.3 Calculate class weights for imbalanced data
print(f"Training set size: {X_train.shape[0]} samples")
print("Calculating class weights for imbalanced data...")


# Train Logistic Regression classifier with class weights
print("Training Logistic Regression classifier with class weights...")
LR_classifier = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 2.4 Train the model
print("\nTraining the binary classification model with class weights...")
LR_classifier.fit(X_train, y_train)

# 2.5 Make predictions on test set
print("Making predictions...")
y_pred = LR_classifier.predict(X_test)

# Save the trained model and all encoders
model_file = f'model_training/models_checkpoint/binary/LogisticRegression_trained_model_{timestamp}.pkl'
with open(model_file, 'wb') as f:
    pickle.dump({
        'model': LR_classifier,
        'label_encoder': le,
        'protocol_encoder': protocol_encoder,
        'state_encoder': state_encoder
    }, f, pickle.HIGHEST_PROTOCOL)

print(f"Trained model and encoders saved to: {os.path.abspath(model_file)}")


# 2.6 Calculate and print metrics
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.4f}")

# Classification report with proper target names for binary classification
target_names = ['BENIGN', 'ATTACK']  # Binary class names
print("\nClassification Report of Logistic Regression:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 2.7 Plot confusion matrix (improved readability)
print("\nPlotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

# Bigger figure (scale by number of classes for flexibility)
n_classes = len(le.classes_)
plt.figure(figsize=(1.2 * n_classes, 1.0 * n_classes))

# Plot with rotated x-labels
disp.plot(xticks_rotation=90, cmap="Blues", values_format="d")

# Adjust label font sizes
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Shrink numbers inside cells
for text in disp.text_.ravel():
    text.set_fontsize(8)

plt.tight_layout()
plt.savefig("model_training/results_binary/LogisticRegression/Logistic_Regression_confusion_matrix.png", dpi=300)  # higher DPI for sharper text
plt.close()

# 2.8 Show class distribution and weights effectiveness
print(f"\nTraining set size: {X_train.shape[0]} samples")


results = pd.DataFrame({
    'true_label': y_test,
    'true_label_name': le.inverse_transform(y_test),
    'predicted_label': y_pred,
    'predicted_label_name': le.inverse_transform(y_pred),
    'correct': (y_test == y_pred)
})

# Save results to CSV
results_file = f'model_training/results_binary/LogisticRegression/LogisticRegression_binary_predictions_{timestamp}.csv'
results.to_csv(results_file, index=False)

# Save class mapping
class_mapping = pd.DataFrame({
    'class_index': range(len(le.classes_)),
    'class_name': le.classes_
})
class_mapping.to_csv('model_training/results_binary/LogisticRegression/class_mapping.csv', index=False)

print(f"\nBinary predictions saved to: {os.path.abspath(results_file)}")
print(f"Class mapping saved to: {os.path.abspath('model_training/results_binary/LogisticRegression/class_mapping.csv')}")

print("\n" + "="*80)
print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
print("="*80)