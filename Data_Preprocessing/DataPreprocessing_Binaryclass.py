# ============================================================================
# NETWORK INTRUSION DETECTION - DATA PREPROCESSING WITH HYBRID FEATURE SELECTION
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style='darkgrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import os
import pickle
from datetime import datetime

# ============================================================================
# SECTION 1: DATA PREPROCESSING
# ============================================================================

# 1.1 Loading the dataset
data = pd.read_csv('data/merged_data/merged_datasets.csv')

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

# --- STEP 2: Feature engineering ---
print(f"\nSTEP 2: Feature engineering...")
print("-" * 40)

# 1. Average packet size (forward + backward combined)
data['avg_pkt_size'] = (data['source_bytes'] + data['dest_bytes']) / \
                       (data['source_pkts'] + data['dest_pkts'] + 1e-6)

# 2. Packet ratio (forward / backward)
data['pkt_ratio'] = data['source_pkts'] / (data['dest_pkts'] + 1)

# 3. Byte ratio (source / destination)
data['byte_ratio'] = data['source_bytes'] / (data['dest_bytes'] + 1)

# 4. Request–response *average packet size* ratio
data['req_resp_avg_pkt_ratio'] = (data['source_bytes'] / (data['source_pkts'] + 1)) / \
                                 (data['dest_bytes'] / (data['dest_pkts'] + 1) + 1e-6)

# 5. Window-to-payload ratio
data['win_payload_ratio'] = (data['tcp_win_fwd'] + data['tcp_win_bwd']) / \
                            (data['source_bytes'] + data['dest_bytes'] + 1)

# 6. Bytes per second (throughput)
data['bytes_per_sec'] = (data['source_bytes'] + data['dest_bytes']) / (data['duration'] + 1)

# 7. Packets per second (packet rate)
data['pkts_per_sec'] = (data['source_pkts'] + data['dest_pkts']) / (data['duration'] + 1)

# Check engineered features
print("Engineered features added:")
for feat in [
    'avg_pkt_size', 'pkt_ratio', 'byte_ratio',
    'req_resp_avg_pkt_ratio', 'win_payload_ratio',
    'bytes_per_sec', 'pkts_per_sec'
]:
    print(f"  {feat}: mean={data[feat].mean():.3f}, std={data[feat].std():.3f}, "
          f"min={data[feat].min():.3f}, max={data[feat].max():.3f}")

# Update final feature list 
final_features = list(data.columns)
print(f"\nTotal features after engineering: {len(final_features)}")


#Binarize Benign vs attack(0 - 1) 
data['label'] = data['label'].apply(lambda x : 1 if x != 'BENIGN' else 0)

# Create label encoder for target variable
label_encoder = LabelEncoder()
label_encoder.fit([0, 1])

# Display basic info about the cleaned data
print('\nData types and non-null counts:')
print(data.info())



# Renaming the columns by removing leading/trailing whitespace
col_names = {col: col.strip() for col in data.columns}
data.rename(columns = col_names, inplace = True)


# Overview of Columns
stats = data.describe().transpose()


print("check on feature label")
print(data['label'].unique())

# distribution of labels
print(data['label'].value_counts())

# ============================================================================
# CATEGORICAL FEATURE ENCODING
# ============================================================================

print("\n" + "="*60)
print("ENCODING CATEGORICAL FEATURES")
print("="*60)

# --- Protocol Column Encoding ---
print("\nEncoding protocol column...")
if 'protocol' in data.columns:
    print(f"Original protocol values: {sorted(data['protocol'].unique())}")
    print(f"Protocol value counts:\n{data['protocol'].value_counts()}")
    
    # Handle missing values in protocol
    if data['protocol'].isnull().sum() > 0:
        print(f"Found {data['protocol'].isnull().sum()} missing values in protocol, filling with 'unknown'")
        data['protocol'] = data['protocol'].fillna('unknown')
    
    # Convert to string to handle mixed types
    data['protocol'] = data['protocol'].astype(str)
    
    # Create and fit protocol encoder
    protocol_encoder = LabelEncoder()
    data['protocol_encoded'] = protocol_encoder.fit_transform(data['protocol'])
    
    print("Protocol encoding mapping:")
    for i, label in enumerate(protocol_encoder.classes_):
        count = (data['protocol_encoded'] == i).sum()
        print(f"  {i}: {label} ({count:,} samples)")
else:
    print("Warning: 'protocol' column not found in dataset")
    protocol_encoder = None

# --- State Column Encoding ---
print("\nEncoding state column...")
if 'state' in data.columns:
    print(f"Original state values: {sorted(data['state'].unique())}")
    print(f"State value counts:\n{data['state'].value_counts()}")
    
    # Handle missing values in state
    if data['state'].isnull().sum() > 0:
        print(f"Found {data['state'].isnull().sum()} missing values in state, filling with 'unknown'")
        data['state'] = data['state'].fillna('unknown')
    
    # Convert to string to handle mixed types
    data['state'] = data['state'].astype(str)
    
    # Create and fit state encoder
    state_encoder = LabelEncoder()
    data['state_encoded'] = state_encoder.fit_transform(data['state'])
    
    print("State encoding mapping:")
    for i, label in enumerate(state_encoder.classes_):
        count = (data['state_encoded'] == i).sum()
        print(f"  {i}: {label} ({count:,} samples)")
else:
    print("Warning: 'state' column not found in dataset")
    state_encoder = None

print("Categorical feature encoding completed!")

# ============================================================================
# HYBRID FEATURE SELECTION PIPELINE
# ============================================================================

print("\n" + "="*80)
print("HYBRID FEATURE SELECTION PIPELINE")
print("="*80)

# Build feature matrix X (numeric + encoded categorical features)
drop_cols = {'label', 'dataset_id'}
numeric_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
X = data[numeric_cols].copy()

print(f"Starting with {X.shape[1]} features")
print(f"Dataset shape: {X.shape}")

# --- STEP 1: Correlation Filter ---
print("\nSTEP 1: Correlation Filter...")
print("-" * 40)

# Calculate Spearman correlation with target
corr = X.apply(lambda col: abs(col.corr(data['label'], method='spearman')))
corr = corr.dropna()

corr_threshold = 0.10
core_features = corr[corr > corr_threshold].index.tolist()

print(f"Features with |p| > {corr_threshold}: {len(core_features)}")
print("Top 10 correlated features:")
top_corr = corr.sort_values(ascending=False).head(10)
for feat, corr_val in top_corr.items():
    print(f"  {feat:<30}: {corr_val:.3f}")

# --- STEP 2: Model-based Importance (Random Forest) ---
print(f"\nSTEP 2: Model-based Importance...")
print("-" * 40)

# Use sampling for large datasets
if X.shape[0] > 100000:
    print("Large dataset detected, using sample for feature importance calculation...")
    sample_size = min(50000, X.shape[0])
    sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = data['label'].iloc[sample_idx]
else:
    X_sample = X
    y_sample = data['label']

print(f"Training Random Forest on {X_sample.shape[0]} samples...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_depth=10,
    min_samples_split=100
)

rf.fit(X_sample, y_sample)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importance_threshold = 0.01
important_features = importances[importances > importance_threshold].index.tolist()

print(f"Features with importance > {importance_threshold}: {len(important_features)}")
print("Top 10 important features:")
top_importance = importances.sort_values(ascending=False).head(10)
for feat, imp_val in top_importance.items():
    print(f"  {feat:<30}: {imp_val:.4f}")

# --- STEP 3: Mutual Information ---
print(f"\nSTEP 3: Mutual Information Analysis...")
print("-" * 40)

print("Computing mutual information scores...")
mi = mutual_info_classif(X_sample, y_sample, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi, index=X.columns)
mi_threshold = 0.01
mi_features = mi_series[mi_series > mi_threshold].index.tolist()

print(f"Features with MI > {mi_threshold}: {len(mi_features)}")
print("Top 10 MI features:")
top_mi = mi_series.sort_values(ascending=False).head(10)
for feat, mi_val in top_mi.items():
    print(f"  {feat:<30}: {mi_val:.4f}")

# --- STEP 4: Combine All Selected Sets ---
print(f"\nSTEP 4: Combining Feature Sets...")
print("-" * 40)

selected_features = list(set(core_features + important_features + mi_features))
print(f"Combined features from all methods: {len(selected_features)}")

# Create summary of selection methods
feature_sources = {}
for feat in selected_features:
    sources = []
    if feat in core_features:
        sources.append("Correlation")
    if feat in important_features:
        sources.append("RF_Importance")
    if feat in mi_features:
        sources.append("Mutual_Info")
    feature_sources[feat] = sources

print("\nFeature selection method summary:")
method_counts = {"Correlation": len(core_features), 
                "RF_Importance": len(important_features), 
                "Mutual_Info": len(mi_features)}
for method, count in method_counts.items():
    print(f"  {method}: {count} features")

# --- STEP 5: Redundancy Check ---
print(f"\nSTEP 5: Redundancy Check...")
print("-" * 40)

if len(selected_features) > 1:
    print("Computing correlation matrix for selected features...")
    X_selected_temp = X[selected_features]
    corr_matrix = X_selected_temp.corr().abs()
    
    # Create upper triangle matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop (correlation > 0.95)
    redundancy_threshold = 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > redundancy_threshold)]
    
    print(f"Features with correlation > {redundancy_threshold}: {len(to_drop)}")
    if to_drop:
        print("Highly correlated features to remove:")
        for feat in to_drop:
            correlated_with = upper[feat][upper[feat] > redundancy_threshold].index.tolist()
            print(f"  {feat} (correlated with: {correlated_with})")
    
    final_features = [f for f in selected_features if f not in to_drop]
else:
    final_features = selected_features
    to_drop = []

print(f"\nFinal features after redundancy check: {len(final_features)}")


# --- STEP 7: Create Final Feature Matrix ---
print(f"\nSTEP 7: Creating Final Feature Matrix...")
print("-" * 40)

X = X[final_features]

# Create detailed feature selection report
print("\n" + "="*60)
print("FEATURE SELECTION SUMMARY REPORT")
print("="*60)

print(f"Original features: {len(numeric_cols)}")
print(f"After correlation filter (|p| > {corr_threshold}): {len(core_features)}")
print(f"After RF importance (> {importance_threshold}): {len(important_features)}")
print(f"After mutual information (> {mi_threshold}): {len(mi_features)}")
print(f"Combined unique features: {len(selected_features)}")
print(f"After redundancy removal (p < {redundancy_threshold}): {len(final_features)}")
print(f"Final feature reduction: {len(numeric_cols)} -> {len(final_features)} ({len(final_features)/len(numeric_cols)*100:.1f}% retained)")

# Show final selected features with their selection methods and scores
print(f"\nFinal Selected Features ({len(final_features)}):")
print("-" * 60)
final_feature_info = []
for feat in final_features:
    info = {
        'feature': feat,
        'correlation': corr.get(feat, 0.0),
        'rf_importance': importances.get(feat, 0.0),
        'mutual_info': mi_series.get(feat, 0.0),
        'methods': ', '.join(feature_sources.get(feat, []))
    }
    final_feature_info.append(info)

# Sort by combined score
final_feature_info.sort(key=lambda x: x['correlation'] + x['rf_importance'] + x['mutual_info'], reverse=True)

for info in final_feature_info:
    print(f"{info['feature']:<25} | Corr: {info['correlation']:.3f} | RF: {info['rf_importance']:.4f} | MI: {info['mutual_info']:.4f} | Methods: {info['methods']}")

print("="*60)

# Update selected_features to final_features for consistency
selected_features = final_features

# ============================================================================
# TRAIN/TEST SPLIT, DOWNSAMPLING, AND SCALING
# ============================================================================

# Split the data (keep original imbalance for evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, data['label'], test_size=0.2, random_state=42, stratify=data['label']
)
print(f"\nDataset split - Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ==============================================
# Downsample Benign class (on training set only)
# ==============================================
print("\nDownsampling majority (Benign) class in training set...")

train_df = pd.concat([X_train, y_train], axis=1)
benign_train = train_df[train_df['label'] == 0]
attack_train = train_df[train_df['label'] == 1]

# Print before-downsampling info
benign_count_before = len(benign_train)
attack_count_before = len(attack_train)
total_before = benign_count_before + attack_count_before
print(f"\nBefore downsampling:")
print(f"  Benign: {benign_count_before:,} ({benign_count_before / total_before * 100:.2f}%)")
print(f"  Attack: {attack_count_before:,} ({attack_count_before / total_before * 100:.2f}%)")
print(f"  Total : {total_before:,}")

# Define desired ratio: 1 attack : 2 benign
target_benign = int(len(attack_train) * 2)
benign_downsampled = benign_train.sample(n=target_benign, random_state=42)

train_balanced = pd.concat([benign_downsampled, attack_train], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

X_train = train_balanced.drop(columns=['label'])
y_train = train_balanced['label']

# Print after-downsampling info
benign_count_after = (y_train == 0).sum()
attack_count_after = (y_train == 1).sum()
total_after = benign_count_after + attack_count_after

print(f"\nAfter downsampling:")
print(f"  Benign: {benign_count_after:,} ({benign_count_after / total_after * 100:.2f}%)")
print(f"  Attack: {attack_count_after:,} ({attack_count_after / total_after * 100:.2f}%)")
print(f"  Total : {total_after:,}")
print(f"  Downsampling reduced training set size by {100 * (1 - total_after / total_before):.1f}%")

print(f"\nTest set remains untouched (real distribution): {y_test.value_counts().to_dict()}")

# ==============================================
# Memory optimization
# ==============================================
print("\nOptimizing memory usage...")
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# ==============================================
# Scaling
# ==============================================
print("\nScaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames with proper column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)


# Save processed data
print("\nSaving processed data...")
os.makedirs('data/processed_data_binary', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save training data
train_data = pd.concat([X_train_scaled, y_train], axis=1)
train_file = f'data/processed_data_binary/X_train_scaled_binary_class_{timestamp}.csv'
train_data.to_csv(train_file, index=False)

# Save test data
test_data = pd.concat([X_test_scaled, y_test], axis=1)
test_file = f'data/processed_data_binary/X_test_scaled_binary_class_{timestamp}.csv'
test_data.to_csv(test_file, index=False)



# Save label encoder
label_encoder_file = f'data/processed_data_binary/label_encoder_binary_class_{timestamp}.pkl'
with open(label_encoder_file, 'wb') as f:
    pickle.dump(label_encoder, f, pickle.HIGHEST_PROTOCOL)

if 'protocol_encoder' in locals() and protocol_encoder is not None:
    protocol_encoder_file = f'data/processed_data_binary/protocol_encoder_binary_class_{timestamp}.pkl'
    with open(protocol_encoder_file, 'wb') as f:
        pickle.dump(protocol_encoder, f, pickle.HIGHEST_PROTOCOL)
    print(f"Protocol encoder saved to: {os.path.abspath(protocol_encoder_file)}")

if 'state_encoder' in locals() and state_encoder is not None:
    state_encoder_file = f'data/processed_data_binary/state_encoder_binary_class_{timestamp}.pkl'
    with open(state_encoder_file, 'wb') as f:
        pickle.dump(state_encoder, f, pickle.HIGHEST_PROTOCOL)
    print(f"State encoder saved to: {os.path.abspath(state_encoder_file)}")

# Save feature selection details
feature_selection_file = f'data/processed_data_binary/feature_selection_report_binary_class_{timestamp}.txt'
with open(feature_selection_file, 'w') as f:
    f.write("HYBRID FEATURE SELECTION REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Original features: {len(numeric_cols)}\n")
    f.write(f"After correlation filter (|p| > {corr_threshold}): {len(core_features)}\n")
    f.write(f"After RF importance (> {importance_threshold}): {len(important_features)}\n")
    f.write(f"After mutual information (> {mi_threshold}): {len(mi_features)}\n")
    f.write(f"Combined unique features: {len(selected_features)}\n")
    f.write(f"After redundancy removal (p < {redundancy_threshold}): {len(final_features)}\n")
    f.write(f"Final feature reduction: {len(numeric_cols)} -> {len(final_features)} ({len(final_features)/len(numeric_cols)*100:.1f}% retained)\n\n")
    
    f.write("FINAL SELECTED FEATURES:\n")
    f.write("-" * 30 + "\n")
    for info in final_feature_info:
        f.write(f"{info['feature']:<25} | Corr: {info['correlation']:.3f} | RF: {info['rf_importance']:.4f} | MI: {info['mutual_info']:.4f} | Methods: {info['methods']}\n")

# Save feature names
feature_file = f'data/processed_data_binary/selected_features_hybrid_{timestamp}.txt'
with open(feature_file, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print(f"Training data saved to: {os.path.abspath(train_file)}")
print(f"Test data saved to: {os.path.abspath(test_file)}")
print(f"Selected features saved to: {os.path.abspath(feature_file)}")
print(f"Feature selection report saved to: {os.path.abspath(feature_selection_file)}")

print("\n" + "="*80)
print("HYBRID FEATURE SELECTION PREPROCESSING COMPLETED")
print("="*80)
print(f"Final training set shape: {X_train.shape}")
print(f"Final test set shape: {X_test.shape}")
print(f"Selected features: {len(selected_features)}")
print("="*80)
