import pandas as pd

ds1_cols = [3, 4, 5, 6, 7, 8, 16, 17, 47]
ds1_paths = ["./sets/dataset_1_UNSW-NB15_1.csv",
       "./sets/dataset_1_UNSW-NB15_2.csv",
       "./sets/dataset_1_UNSW-NB15_3.csv",
       "./sets/dataset_1_UNSW-NB15_4.csv"]


ds1_csvs = [pd.read_csv(file, header=None, usecols=ds1_cols) for file in ds1_paths]

merged_ds1 = pd.concat(ds1_csvs, ignore_index=True)
merged_ds1.columns = ["destination_port", "protocol", "state", "duration", "source_bytes", "dest_bytes", "source_pkts", "dest_pkts", "label"]

# merged_ds1['duration'] = merged_ds1['duration'] * 1_000_000
merged_ds1['label'] = merged_ds1['label'].replace('', pd.NA).fillna('BENIGN')
merged_ds1['dataset_id'] = 0

merged_ds1.to_csv("merged_unsw_nb15.csv", index=False, header=True)