import pandas as pd

print("=" * 80)
print("UNSW-NB15 DATASET PROCESSING WITH DUPLICATE REMOVAL")
print("=" * 80)

# Column indices (zero-based) for UNSW-NB15 CSVs
ds1_cols = [3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 22, 23, 47]
ds1_paths = ["network-intrusion-dataset/UNSW_NB15/UNSW-NB15_1.csv",
       "network-intrusion-dataset/UNSW_NB15/UNSW-NB15_2.csv",
       "network-intrusion-dataset/UNSW_NB15/UNSW-NB15_3.csv",
       "network-intrusion-dataset/UNSW_NB15/UNSW-NB15_4.csv"]

print(f"Processing {len(ds1_paths)} UNSW-NB15 files...")
ds1_csvs = []
total_rows_before = 0

for i, file_path in enumerate(ds1_paths, 1):
    print(f"\nProcessing file {i}: {file_path.split('/')[-1]}")
    df_part = pd.read_csv(file_path, header=None, usecols=ds1_cols)
    rows_loaded = len(df_part)
    total_rows_before += rows_loaded
    print(f"  Rows loaded: {rows_loaded:,}")
    ds1_csvs.append(df_part)

print(f"\nConcatenating {len(ds1_csvs)} files...")
merged_ds1 = pd.concat(ds1_csvs, ignore_index=True)
print(f"Total rows before duplicate removal: {len(merged_ds1):,}")

# Assign canonical column names (dataset_id is added later)
merged_ds1.columns = [
    "destination_port",
    "protocol",
    "state",
    "duration",
    "source_bytes",
    "dest_bytes",
    "source_pkts",
    "dest_pkts",
    "tcp_win_fwd",
    "tcp_win_bwd",
    "mean_seg_size_fwd",
    "mean_seg_size_bwd",
    "label",
]

print("\nProcessing labels and adding dataset_id...")
# merged_ds1['duration'] = merged_ds1['duration'] * 1_000_000
merged_ds1['label'] = merged_ds1['label'].replace('', pd.NA).fillna('BENIGN')
merged_ds1['dataset_id'] = 0

# Remove duplicates
DUPLICATE_CHECK_COLS = [col for col in merged_ds1.columns if col != 'dataset_id']
print(f"\nChecking for duplicates based on {len(DUPLICATE_CHECK_COLS)} columns...")

rows_before_dedup = len(merged_ds1)
merged_ds1_clean = merged_ds1.drop_duplicates(subset=DUPLICATE_CHECK_COLS, keep='first')
rows_after_dedup = len(merged_ds1_clean)
duplicates_removed = rows_before_dedup - rows_after_dedup

print(f"\nDUPLICATE REMOVAL RESULTS:")
print(f"  Rows before: {rows_before_dedup:,}")
print(f"  Rows after:  {rows_after_dedup:,}")
print(f"  Duplicates removed: {duplicates_removed:,}")
print(f"  Duplicate percentage: {(duplicates_removed/rows_before_dedup)*100:.2f}%")

print(f"\nAttack type distribution:")
attack_counts = merged_ds1_clean['label'].value_counts()
for attack_type, count in attack_counts.items():
    percentage = (count / len(merged_ds1_clean)) * 100
    print(f"  {attack_type}: {count:,} ({percentage:.2f}%)")

# Re-order columns so they match the CIC 2017 processed file exactly
export_cols = [
    "destination_port",
    "protocol",
    "state",
    "duration",
    "source_bytes",
    "dest_bytes",
    "source_pkts",
    "dest_pkts",
    "mean_seg_size_fwd",
    "mean_seg_size_bwd",
    "tcp_win_fwd",
    "tcp_win_bwd",
    "label",
    "dataset_id",
]

# Save with index
merged_ds1_final = merged_ds1_clean[export_cols].reset_index(drop=True)
output_path = "processed_data/merged_unsw_nb15.csv"
merged_ds1_final.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")
print(f"Index column added: Yes (0-based integer index)")
print(f"Final shape: {merged_ds1_final.shape}")
print("=" * 80)