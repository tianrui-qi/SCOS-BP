import torch
import numpy as np
import pandas as pd

import os
import scipy.io


# load data
icloud_path = "/Users/tianrui.qi/Library/Mobile Documents/com~apple~CloudDocs/"
data_path = os.path.join(icloud_path, 'Research/RoblyerLab/data/waveform/')
data = scipy.io.loadmat(os.path.join(data_path, 'raw.mat'))["data_store"][0, 0]

# create profile.csv
def scalarize(arr, dtype=None):
    out = [
        np.asarray(x).ravel()[0] if np.asarray(x).size else np.nan for x in arr
    ]
    return np.array(out, dtype=dtype) if dtype is not None else np.array(out)

df = pd.DataFrame({
    'id'        : scalarize(data[2], dtype=str).flatten(),
    'group'     : scalarize(data[3], dtype=str).flatten(),
    'repeat'    : scalarize(data[4], dtype=bool).flatten(),
    'condition' : scalarize(data[5], dtype=int).flatten()
})
df["health"] = df["group"] != "hypertensive"
df["system"] = df["group"] != "original"
df = df[['id', 'health', 'system', 'repeat', 'condition']]
# for all sample that are repeated, learn it be test data
df["pretrain"] = np.where(df["repeat"], 2, 0)
# for each subject and each condition, use 10% of sample as validation data
for _, group in df[~df["repeat"]].groupby(['id', 'condition']):
    df.loc[group.sample(frac=0.1, random_state=0).index, "pretrain"] = 1
# save
df.to_csv(os.path.join(data_path, 'profile.csv'), index=False)

# store data[0] and data[1] as torch tensors
torch.save(
    torch.from_numpy(data[0]).to(torch.float).transpose(-1, -2), 
    os.path.join(data_path, 'x.pt')
)
torch.save(
    torch.from_numpy(data[1]).to(torch.float), 
    os.path.join(data_path, 'y.pt')
)