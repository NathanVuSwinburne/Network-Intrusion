import pandas as pd

# Column indices (zero-based) for UNSW-NB15 CSVs
#  3  → dsport                (destination_port)
#  4  → proto                 (protocol)
#  5  → state
#  6  → dur                   (duration)
#  7  → sbytes                (source_bytes)
#  8  → dbytes                (dest_bytes)
# 16 → Spkts                 (source_pkts)
# 17 → Dpkts                 (dest_pkts)
# 18 → swin                  (tcp_win_fwd)
# 19 → dwin                  (tcp_win_bwd)
# 22 → smeansz               (mean_seg_size_fwd)
# 23 → dmeansz               (mean_seg_size_bwd)
# 47 → Label                 (binary label; 0 benign, 1 attack)
ds1_cols = [3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 22, 23, 47]
ds1_paths = ["network-intrusion-dataset/UNSW_NB15/UNSW-NB15_1.csv",
       "network-intrusion-dataset/UNSW_NB15/UNSW-NB15_2.csv",
       "network-intrusion-dataset/UNSW_NB15/UNSW-NB15_3.csv",
       "network-intrusion-dataset/UNSW_NB15/UNSW-NB15_4.csv"]

ds1_csvs = [pd.read_csv(file, header=None, usecols=ds1_cols) for file in ds1_paths]

merged_ds1 = pd.concat(ds1_csvs, ignore_index=True)
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

# merged_ds1['duration'] = merged_ds1['duration'] * 1_000_000
merged_ds1['label'] = merged_ds1['label'].replace('', pd.NA).fillna('BENIGN')
merged_ds1['dataset_id'] = 0

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
merged_ds1[export_cols].to_csv("processed_data/merged_unsw_nb15.csv", index=False)