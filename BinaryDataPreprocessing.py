
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
# Loading the dataset
data1 = pd.read_csv('C:/Users/Admin/PycharmProjects/Network Intrusion/network-intrusion-dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
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

# --- Add binary target ---
data['y_binary'] = np.where(data['Attack Type'] == 'BENIGN', 0, 1)

# Quick check
print("\nBinary target distribution (y_binary):")
print(data['y_binary'].value_counts())

# --- 1) Build feature matrix X (numeric only; drop label columns) -------------
drop_cols = {'Attack Type', 'y_binary', 'y_multi'}
feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
X = data[feature_cols].copy()

# --- 2) BINARY FEATURE RANKING (Pearson corr with y_binary) -------------------
y_bin = data['y_binary']
corr_bin = X.corrwith(y_bin, method='pearson').sort_values(ascending=False)
corr_bin_df = corr_bin.rename('Abs Corr (Binary)').abs().sort_values(ascending=False).reset_index()
corr_bin_df.columns = ['Feature', 'Abs Corr (Binary)']

print("\n=== Top 20 features for BINARY (Benign=0, Attack=1) by |Pearson correlation| ===")
print(corr_bin_df.head(20).to_string(index=False))

# ONLY positive correlations:
pos_bin = corr_bin[corr_bin >= 0.01].sort_values(ascending=False)
print(f"\n# Positive-correlation features with y_binary (count={pos_bin.shape[0]}):")
for i, (feat, val) in enumerate(pos_bin.items(), start=1):
    print(f"{i:>2}. {feat:<30} : {val:.2f}")

# Keep only positively correlated features
X = X[pos_bin.index]
print(f"\nKeeping {len(pos_bin)} features with positive correlation >= 0.01")

# Split the data before scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)
print(f"\nDataset split - Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Initialize and fit the scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to test data (without fitting)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain column names
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Memory optimization
print("\nOptimizing memory usage...")
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Apply SMOTE to balance the training set only
print("\nApplying SMOTE to balance the training set...")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Training set before SMOTE: {X_train.shape[0]} samples")
print(f"Training set after SMOTE: {X_train_balanced.shape[0]} samples")
print("Class distribution after SMOTE:", pd.Series(y_train_balanced).value_counts().to_dict())

# Apply PCA to the balanced training set
print("\nApplying PCA to the balanced training set...")
from sklearn.decomposition import PCA

# Use 95% of variance or max 50 components, whichever is more restrictive
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test)  # Apply same transformation to test set

print(f"Original number of features: {X_train_balanced.shape[1]}")
print(f"Reduced number of components: {pca.n_components_}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")

# Convert back to DataFrames
X_train_final = pd.DataFrame(X_train_pca, index=X_train_balanced.index,
                           columns=[f'PC{i+1}' for i in range(pca.n_components_)])
X_test_final = pd.DataFrame(X_test_pca, index=X_test.index,
                          columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# Create the directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save the processed data
train_data = pd.concat([X_train_final, y_train_balanced], axis=1)
test_data = pd.concat([X_test_final, y_test], axis=1)

train_file = f'processed_data/train_data_balanced_pca_{timestamp}.csv'
test_file = f'processed_data/test_data_pca_{timestamp}.csv'

train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print(f"\nBalanced training data with PCA saved to: {os.path.abspath(train_file)}")
print(f"Test data with PCA saved to: {os.path.abspath(test_file)}")
print("\nData preprocessing completed successfully!")
