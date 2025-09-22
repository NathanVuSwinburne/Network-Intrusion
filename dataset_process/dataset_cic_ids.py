import pandas as pd

# this dataset has some weird column names with whitespaces at the start
cols = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Label",
        " RST Flag Count", "FIN Flag Count", " ACK Flag Count", " SYN Flag Count"]
paths = [
    "./sets/dataset_2_0.csv",
    "./sets/dataset_2_1.csv",
    "./sets/dataset_2_2.csv",
    "./sets/dataset_2_3.csv",
    "./sets/dataset_2_4.csv",
    "./sets/dataset_2_5.csv",
    "./sets/dataset_2_6.csv",
    "./sets/dataset_2_7.csv"
]

csvs = [pd.read_csv(file, usecols=cols) for file in paths]

merged_ds2 = pd.concat(csvs, ignore_index=True)

def infer_protocol(row):
    # infer the protocol from existing information
    if row[' SYN Flag Count'] > 0 or row[' ACK Flag Count'] > 0:
        return 'tcp'
    elif row[' SYN Flag Count'] == 0 and row[' ACK Flag Count'] == 0:
        if row[' Destination Port'] in {53, 123, 161, 67, 68}:
            return 'udp'
        elif row[' Destination Port'] in {80, 443, 22, 21}:
            return 'tcp'
        else:
            return 'udp'  # assume UDP if no TCP flags
    else:
        return 'unknown'

def infer_state(row):
    # best attempt at inferring state on existing information
    syn = row.get('SYN Flag Count', 0)
    ack = row.get('ACK Flag Count', 0)
    fin = row.get('FIN Flag Count', 0)
    rst = row.get('RST Flag Count', 0)
    fwd = row.get('Total Fwd Packets', 0)
    bwd = row.get('Total Backward Packets', 0)

    if rst > 0:
        return 'RST'
    elif syn > 0 and ack == 0 and fin == 0:
        return 'INT'  # SYN sent, no response yet
    elif syn > 0 and ack > 0 and fin == 0:
        return 'CON'  # handshake complete
    elif fin > 0:
        return 'FIN'  # finished
    elif fwd > 0 and bwd == 0:
        return 'REQ'  # only request seen
    elif bwd > 0 and fwd == 0:
        return 'ACC'  # only response seen (safter timeout)
    elif fwd > 0 and bwd > 0:
        return 'TXD'  # data exchange in both directions
    else:
        return 'UNKNOWN'  # unknown

merged_ds2['state'] = merged_ds2.apply(infer_state, axis=1)
merged_ds2['protocol'] = merged_ds2.apply(infer_protocol, axis=1)



# rename the columns. a, b, c, d are columns included but not needed and are removed when the csv is exported so they dont really need a name lol
merged_ds2.columns = ["destination_port", "duration", "source_pkts", "dest_pkts", "source_bytes", "dest_bytes","a", "b", "c","d", "label", "state", "protocol"]

# convert microseconds into seconds, after the rename
merged_ds2['duration'] = merged_ds2['duration'] / 1_000_000
merged_ds2['dataset_id'] = 1

export_cols = ["destination_port", "protocol", "state", "duration", "source_bytes", "dest_bytes", "source_pkts", "dest_pkts", "label", "dataset_id"]
merged_ds2[export_cols].to_csv("merged_cic_ids_2017.csv", index=False, header=True)
