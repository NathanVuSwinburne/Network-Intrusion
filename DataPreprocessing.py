
# ============================================================================
# NETWORK INTRUSION DETECTION - DATA PREPROCESSING AND MODEL TRAINING
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import os
from datetime import datetime

# ============================================================================
# SECTION 1: DATA PREPROCESSING
# ============================================================================

# 1.1 Loading the dataset
data1 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Monday-WorkingHours.pcap_ISCX.csv')
data2 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Tuesday-WorkingHours.pcap_ISCX.csv')
data3 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Wednesday-workingHours.pcap_ISCX.csv')
data4 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
data5 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
data6 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv')
data7 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
data8 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')


data_list = [data1, data2, data3, data4, data5, data6, data7, data8]

# print('Data dimensions: ')
# for i, data in enumerate(data_list, start = 1):
#   rows, cols = data.shape
#   print(f'Data{i} -> {rows} rows, {cols} columns')

# Data dimensions:
# Data1 -> 225745 rows, 79 columns
# Data2 -> 445909 rows, 79 columns
# Data3 -> 692703 rows, 79 columns
# Data4 -> 170366 rows, 79 columns
# Data5 -> 288602 rows, 79 columns
# Data6 -> 191033 rows, 79 columns
# Data7 -> 286467 rows, 79 columns
# Data8 -> 225745 rows, 79 columns


data = pd.concat(data_list)

# Drop duplicate rows
print("Dropping duplicate rows...")
initial_rows = len(data)
data = data.drop_duplicates()
dropped_rows = initial_rows - len(data)
print(f"Dropped {dropped_rows} duplicate rows")

# Replace infinite values with NaN
print("Handling infinite values...")
data = data.replace([np.inf, -np.inf], np.nan)

print("Total missing value:")
missing = data.isna().sum()
print(missing.loc[missing > 0])
# Total missing value:
# Flow Bytes/s       1257
#  Flow Packets/s    1257
# dtype: int64

# # Calculating missing value percentage in the dataset
# mis_per = (missing / len(data)) * 100
# mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
# mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})
#
# print(mis_table.loc[mis_per > 0])
#                  Missing Values  Percentage of Total Values
# Flow Bytes/s               1257                        0.06
#  Flow Packets/s            1257                        0.06

# Fill missing values with median (for numeric columns only)
print("Filling missing values with median...")
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        before = data[col].isnull().sum()
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
        print(f"Filled {before} missing values in {col} with median: {median_val:.2f}")

rows, cols = data.shape
print('\nFinal dataset after cleaning:')
print(f"Number of rows: {rows:,}")
print(f"Number of columns: {cols}")
print(f"Number of duplicate rows removed: {dropped_rows:,}")
print(f"Total missing values after cleaning: {data.isnull().sum().sum()}")


# # Display basic info about the cleaned data
# print('\nData types and non-null counts:')
# print(data.info())


# Renaming the columns by removing leading/trailing whitespace
# We remove leading/trailing whitespace to avoid duplicate-looking columns/labels,
# prevent bugs in merges and model training, and make code more consistent and reliable.
col_names = {col: col.strip() for col in data.columns}
data.rename(columns = col_names, inplace = True)


# Overview of Columns
stats = data.describe().transpose()

print("All type of attacks:")
print(data['Label'].unique())

# Types of attacks & normal instances (BENIGN)
print(data['Label'].value_counts())

# Creating a dictionary that maps each label to its attack type
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

# Value normalization by creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
data['Attack Type'] = data['Label'].map(attack_map)

print(data['Attack Type'].value_counts())


data.drop('Label', axis = 1, inplace = True)


# --- Target encoding (multi-class) ---
le = LabelEncoder()
data['y_multi'] = le.fit_transform(data['Attack Type'])

# Print mapping for clarity
print("Multi-class mapping (y_multi):")
for val in sorted(data['y_multi'].unique()):
    print(f"{val}: {le.inverse_transform([val])[0]}")

# Quick check
print("\nClass distribution (y_multi):")
print(data['y_multi'].value_counts())

# --- 1) Build feature matrix X (numeric only; drop label columns) -------------
drop_cols = {'Attack Type', 'y_multi'}
numeric_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
X = data[numeric_cols].copy()

# Remove constant features (features with zero variance)
print("\nRemoving constant features...")
initial_features = X.shape[1]
constant_features = X.columns[X.var() == 0].tolist()
if constant_features:
    print(f"Found {len(constant_features)} constant features: {constant_features[:5]}{'...' if len(constant_features) > 5 else ''}")
    X = X.drop(columns=constant_features)
    print(f"Removed {len(constant_features)} constant features")

# Calculate correlation with the target variable
print("Calculating feature correlations...")
target_corr = X.apply(lambda x: abs(x.corr(data['y_multi'], method='spearman')))

# Remove features with NaN correlation (shouldn't happen after removing constants, but just in case)
target_corr = target_corr.dropna()

# Select features with correlation > 0.10
selected_features = target_corr[target_corr > 0.10].index.tolist()
X = X[selected_features]

print(f"\nOriginal number of numeric features: {len(numeric_cols)}")
print(f"Number of features with correlation > 0.10: {len(selected_features)}")
print("\nSelected features and their correlation with the target:")
for feat, corr in sorted(zip(selected_features, target_corr[selected_features]), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feat:<30}: {corr:.3f}")

# 1.8 Split the data before scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, data['y_multi'], test_size=0.2, random_state=42, stratify=data['y_multi']
)
print(f"\nDataset split - Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# 1.9 Memory optimization
print("\nOptimizing memory usage...")
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# 1.10 Save processed data
print("\nSaving processed data...")
os.makedirs('processed_data', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save training data
train_data = pd.concat([X_train, y_train], axis=1)
train_file = f'processed_data/X_train_{timestamp}.csv'
train_data.to_csv(train_file, index=False)

# Save test data
test_data = pd.concat([X_test, y_test], axis=1)
test_file = f'processed_data/X_test_{timestamp}.csv'
test_data.to_csv(test_file, index=False)

# Save label encoder
import pickle
le_file = f'processed_data/label_encoder_{timestamp}.pkl'
with open(le_file, 'wb') as f:
    pickle.dump(le, f)

# Save feature names
feature_file = f'processed_data/selected_features_{timestamp}.txt'
with open(feature_file, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print(f"Training data saved to: {os.path.abspath(train_file)}")
print(f"Test data saved to: {os.path.abspath(test_file)}")
print(f"Label encoder saved to: {os.path.abspath(le_file)}")
print(f"Selected features saved to: {os.path.abspath(feature_file)}")

print("\n" + "="*80)
print("DATA PREPROCESSING COMPLETED")
print("="*80)
print(f"Final training set shape: {X_train.shape}")
print(f"Final test set shape: {X_test.shape}")
print(f"Number of classes: {len(le.classes_)}")
print(f"Selected features: {len(selected_features)}")
print("="*80)


