
import pandas as pd
import numpy as np
import os
import glob
import pickle
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create the directory if it doesn't exist
os.makedirs('results_binary/DecisionTree', exist_ok=True)

# 2.2 Load the latest processed data
print("Loading processed data...")
train_files = glob.glob('data/processed_data_binary/X_train_binary_class_*.csv')
test_files = glob.glob('data/processed_data_binary/X_test_binary_class_*.csv')
le_files = glob.glob('data/processed_data_binary/label_encoder_binary_class_*.pkl')
protocol_encoder_files = glob.glob('data/processed_data_binary/protocol_encoder_binary_class_*.pkl')
state_encoder_files = glob.glob('data/processed_data_binary/state_encoder_binary_class_*.pkl')

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


# 2.3 Calculate class weights for imbalanced data
print(f"Training set size: {X_train.shape[0]} samples")
print("Calculating class weights for imbalanced data...")


# Apply scaling
print("\nScaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

