import torch
import scipy.io

import src


waveform = scipy.io.loadmat(
    "C:/Users/tianrui/iCloudDrive/Research/RoblyerLab/data/waveform/raw.mat"
)["data_store"][0, 0]

x = waveform[0]
y = waveform[1]
subject_ids = waveform[2]
groups = waveform[3]
repeats = waveform[4]
conditions = waveform[5]

print(subject_ids)
print(groups)
print(x.shape)
print(y.shape)

x = torch.from_numpy(x[0:16]).permute(0, 2, 1).float()
# print(x.shape)
# # plot x
# plt.plot(x[0, 0, :])
# plt.plot(x[0, 1, :])
# plt.plot(x[0, 2, :])
# plt.show()


D = 128
C_max = 8
L_max = 200
S = 40
stride = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = src.model.SCOST(
    D=D, S=S, stride=stride, C_max=C_max, L_max=L_max,
    num_layers=4, nhead=8, dim_feedforward=512, out_dim=2,
).to(device=device)
model.train()

x = x.to(device=device)
channel_idx = torch.arange(
    x.shape[1], device=x.device
)[None, :].repeat(x.shape[0], 1)

out, token, mlm_mask = model(x, channel_idx)

model.eval()
out = model(x, channel_idx)
print(out.shape)
