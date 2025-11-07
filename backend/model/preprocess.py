import numpy as np
import pandas as pd
import pickle

# Load all artifacts at startup
with open("backend/model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("backend/model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("backend/model/protocol_encoder.pkl", "rb") as f:
    protocol_encoder = pickle.load(f)
with open("backend/model/state_encoder.pkl", "rb") as f:
    state_encoder = pickle.load(f)

selected_features = [line.strip() for line in open("backend/model/selected_features.txt")]

def preprocess_input(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    # --- Feature engineering ---
    df['avg_pkt_size'] = (df['source_bytes'] + df['dest_bytes']) / (df['source_pkts'] + df['dest_pkts'] + 1e-6)
    df['pkt_ratio'] = df['source_pkts'] / (df['dest_pkts'] + 1)
    df['byte_ratio'] = df['source_bytes'] / (df['dest_bytes'] + 1)
    df['req_resp_avg_pkt_ratio'] = (df['source_bytes'] / (df['source_pkts'] + 1)) / \
                                   (df['dest_bytes'] / (df['dest_pkts'] + 1) + 1e-6)
    df['win_payload_ratio'] = (df['tcp_win_fwd'] + df['tcp_win_bwd']) / (df['source_bytes'] + df['dest_bytes'] + 1)
    df['bytes_per_sec'] = (df['source_bytes'] + df['dest_bytes']) / (df['duration'] + 1)
    df['pkts_per_sec'] = (df['source_pkts'] + df['dest_pkts']) / (df['duration'] + 1)

    # --- Encode categoricals ---
    df['protocol_encoded'] = protocol_encoder.transform(df['protocol'].astype(str))
    df['state_encoded'] = state_encoder.transform(df['state'].astype(str))

    # --- Drop helper columns ---
    df = df.drop(columns=['protocol', 'state'], errors='ignore')

    # --- Ensure correct feature order ---
    feature_order = [
        'mean_seg_size_fwd',
        'source_bytes',
        'dest_bytes',
        'tcp_win_fwd',
        'state_encoded',
        'win_payload_ratio',
        'pkt_ratio',
        'tcp_win_bwd',
        'avg_pkt_size',
        'bytes_per_sec',
        'req_resp_avg_pkt_ratio',
        'protocol_encoded',
        'mean_seg_size_bwd',
        'pkts_per_sec',
        'byte_ratio',
        'duration'
    ]
    X = df[feature_order].astype(np.float32)

    # --- Scale ---
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_order)
    return X_scaled

