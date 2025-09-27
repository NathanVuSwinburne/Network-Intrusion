import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Map original (trimmed) CIC-IDS-2017 column names  ➜  canonical column names
RAW_TO_CANON = {
    "Destination Port": "destination_port",
    "Flow Duration": "duration",
    "Total Fwd Packets": "source_pkts",
    "Total Backward Packets": "dest_pkts",
    "Total Length of Fwd Packets": "source_bytes",
    "Total Length of Bwd Packets": "dest_bytes",
    "Avg Fwd Segment Size": "mean_seg_size_fwd",
    "Avg Bwd Segment Size": "mean_seg_size_bwd",
    "Init_Win_bytes_forward": "tcp_win_fwd",
    "Init_Win_bytes_backward": "tcp_win_bwd",
    "Label": "label",
    "RST Flag Count": "rst_flag_cnt",
    "FIN Flag Count": "fin_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
}

USECOLS = list(RAW_TO_CANON.keys())
DATASET_DIR = Path("network-intrusion-dataset/CIC_IDS_2017")
CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

print("=" * 80)
print("CIC-IDS-2017 DATASET PROCESSING WITH DUPLICATE REMOVAL")
print("=" * 80)

# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
frames = []
total_rows_before = 0

for fname in CSV_FILES:
    path = DATASET_DIR / fname
    print(f"\nProcessing: {fname}")
    
    df_part = pd.read_csv(path, usecols=USECOLS, skipinitialspace=True)
    # Strip any residual whitespace in header names (safety)
    df_part.columns = df_part.columns.str.strip()
    
    rows_before = len(df_part)
    total_rows_before += rows_before
    print(f"  Rows loaded: {rows_before:,}")
    
    frames.append(df_part)

# Concatenate all daily CSVs
print(f"\nConcatenating {len(CSV_FILES)} files...")
cic_df = pd.concat(frames, ignore_index=True)
print(f"Total rows before duplicate removal: {len(cic_df):,}")

# Standardise column names -----------------------------------------------------
cic_df.rename(columns=RAW_TO_CANON, inplace=True)

# -----------------------------------------------------------------------------
# Helper functions (operate on canonical names)
# -----------------------------------------------------------------------------

def infer_protocol(row: pd.Series) -> str:
    """Very coarse protocol inference from flag activity and port."""
    if row["syn_flag_cnt"] > 0 or row["ack_flag_cnt"] > 0:
        return "tcp"

    # When no TCP flags are set, make an educated guess from the destination port
    if row["destination_port"] in {53, 123, 161, 67, 68}:
        return "udp"
    if row["destination_port"] in {80, 443, 22, 21}:
        return "tcp"
    return "udp"  # default


def infer_state(row: pd.Series) -> str:
    syn = row["syn_flag_cnt"]
    ack = row["ack_flag_cnt"]
    fin = row["fin_flag_cnt"]
    rst = row["rst_flag_cnt"]
    fwd = row["source_pkts"]
    bwd = row["dest_pkts"]

    if rst > 0:
        return "RST"
    if syn > 0 and ack == 0 and fin == 0:
        return "INT"  # SYN sent, no response yet
    if syn > 0 and ack > 0 and fin == 0:
        return "CON"  # handshake complete
    if fin > 0:
        return "FIN"  # teardown seen
    if fwd > 0 and bwd == 0:
        return "REQ"  # request only
    if bwd > 0 and fwd == 0:
        return "ACC"  # response only
    if fwd > 0 and bwd > 0:
        return "TXD"  # bidirectional data transfer
    return "UNKNOWN"

# -----------------------------------------------------------------------------
# Derived columns & duplicate removal
# -----------------------------------------------------------------------------
print("\nGenerating derived columns...")
cic_df["state"] = cic_df.apply(infer_state, axis=1)
cic_df["protocol"] = cic_df.apply(infer_protocol, axis=1)

# Convert µs  s
cic_df["duration"] = cic_df["duration"] / 1_000_000
cic_df["dataset_id"] = 1

# Define columns for duplicate detection (exclude dataset_id)
DUPLICATE_CHECK_COLS = [
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
    "label"
]

print(f"\nChecking for duplicates based on {len(DUPLICATE_CHECK_COLS)} columns...")
print("Columns used for duplicate detection:")
for col in DUPLICATE_CHECK_COLS:
    print(f"  - {col}")

# Remove duplicates
rows_before_dedup = len(cic_df)
cic_df_clean = cic_df.drop_duplicates(subset=DUPLICATE_CHECK_COLS, keep='first')
rows_after_dedup = len(cic_df_clean)
duplicates_removed = rows_before_dedup - rows_after_dedup

print(f"\nDUPLICATE REMOVAL RESULTS:")
print(f"  Rows before: {rows_before_dedup:,}")
print(f"  Rows after:  {rows_after_dedup:,}")
print(f"  Duplicates removed: {duplicates_removed:,}")
print(f"  Duplicate percentage: {(duplicates_removed/rows_before_dedup)*100:.2f}%")

# -----------------------------------------------------------------------------
# Final dataset statistics
# -----------------------------------------------------------------------------
print(f"\nFINAL DATASET STATISTICS:")
print(f"  Total unique rows: {len(cic_df_clean):,}")
print(f"  Attack type distribution:")
attack_counts = cic_df_clean['label'].value_counts()
for attack_type, count in attack_counts.items():
    percentage = (count / len(cic_df_clean)) * 100
    print(f"    {attack_type}: {count:,} ({percentage:.2f}%)")

EXPORT_COLS = [
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

# Save with index (reset index to create a clean 0-based index)
cic_df_final = cic_df_clean[EXPORT_COLS].reset_index(drop=True)
output_path = "processed_data/merged_cic_ids_2017.csv"
cic_df_final.to_csv(output_path, index=False, header=True)

print(f"\nDataset saved to: {output_path}")
print(f"Index column added: Yes (0-based integer index)")
print(f"Final shape: {cic_df_final.shape}")
print("\n" + "=" * 80)
