import pandas as pd

dset1 = pd.read_csv("./merged_cic_ids_2017.csv")
dset2 = pd.read_csv("./merged_unsw_nb15.csv")

merged_set = pd.concat([dset1, dset2], ignore_index=True)
merged_set.to_csv('merged_datasets.csv', index=False)
