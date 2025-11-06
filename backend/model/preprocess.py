import numpy as np
import pandas as pd
import pickle

# Load all artifacts at startup
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("model/protocol_encoder.pkl", "rb") as f:
    protocol_encoder = pickle.load(f)
with open("model/state_encoder.pkl", "rb") as f:
    state_encoder = pickle.load(f)

selected_features = [line.strip() for line in open("model/selected_features.txt")]

def preprocess_input(user_input: dict) -> np.ndarray:
    """Convert user JSON input into the same processed format used in training."""
    df = pd.DataFrame([user_input])

    # ---- apply same feature engineering ----
    df['avg_pkt_size'] = (df['source_bytes'] + df['dest_bytes']) / (df['source_pkts'] + df['dest_pkts'] + 1e-6)
    df['pkt_ratio'] = df['source_pkts'] / (df['dest_pkts'] + 1)
    df['byte_ratio'] = df['source_bytes'] / (df['dest_bytes'] + 1)
    df['req_resp_avg_pkt_ratio'] = (df['source_bytes'] / (df['source_pkts'] + 1)) / \
                                   (df['dest_bytes'] / (df['dest_pkts'] + 1) + 1e-6)
    df['win_payload_ratio'] = (df['tcp_win_fwd'] + df['tcp_win_bwd']) / (df['source_bytes'] + df['dest_bytes'] + 1)
    df['bytes_per_sec'] = (df['source_bytes'] + df['dest_bytes']) / (df['duration'] + 1)
    df['pkts_per_sec'] = (df['source_pkts'] + df['dest_pkts']) / (df['duration'] + 1)

    # ---- categorical encoders ----
    df['protocol_encoded'] = protocol_encoder.transform(df['protocol'].astype(str))
    df['state_encoded'] = state_encoder.transform(df['state'].astype(str))

    # ---- select the same final features ----
    X = df[selected_features].astype(np.float32)

    # ---- scale ----
    X_scaled = scaler.transform(X)

    return X_scaled
