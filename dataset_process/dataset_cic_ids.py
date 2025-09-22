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

# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
frames = []
for fname in CSV_FILES:
    path = DATASET_DIR / fname
    df_part = pd.read_csv(path, usecols=USECOLS, skipinitialspace=True)
    # Strip any residual whitespace in header names (safety)
    df_part.columns = df_part.columns.str.strip()
    frames.append(df_part)

# Concatenate all daily CSVs
cic_df = pd.concat(frames, ignore_index=True)

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
# Derived columns & final tidy-up
# -----------------------------------------------------------------------------
cic_df["state"] = cic_df.apply(infer_state, axis=1)
cic_df["protocol"] = cic_df.apply(infer_protocol, axis=1)

# Convert µs ➜ s
cic_df["duration"] = cic_df["duration"] / 1_000_000
cic_df["dataset_id"] = 1

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

cic_df[EXPORT_COLS].to_csv("processed_data/merged_cic_ids_2017.csv", index=False, header=True)
