import torch
import lightning

import warnings
import dataclasses
import matplotlib.pyplot as plt

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)
# disable MPS UserWarning: The operator 'aten::col2im' is not currently 
# supported on the MPS backend
warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")


def main():
    step = 1000
    # config
    config = src.config.PretrainT()
    config.analysis()                   # turn off all augmentations
    config.data.filter_level = "All"    # mask sure all channels are valid
    config.data.batch_size = 32
    # device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    # data, fixed at first batch
    dm = src.data.Module(**dataclasses.asdict(config.data))
    dm.setup()
    x, channel_idx, _ = next(iter(dm.train_dataloader()))
    x, channel_idx = x.to(device), channel_idx.to(device)
    # model
    model = src.model.SCOST(**dataclasses.asdict(config.model)).to(device)
    # train
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.runner.lr)
    for s in range(step):
        opt.zero_grad(set_to_none=True)
        pred, token = model.forwardReconstruction(x, channel_idx)
        loss = torch.nn.functional.smooth_l1_loss(pred, token)
        loss.backward()
        opt.step()
        if s % 50 != 0 and s != step - 1: continue
        print(" | ".join([
            f"step={s:04d}", 
            f"loss={float(loss):.6f}",
            f"#masked={int(len(token))}",
        ]))
    # eval
    model.eval()
    with torch.no_grad():
        x, y = model.forwardReconstruction(x, channel_idx, user_mask=3)
    plt.subplot(3, 1, 1)
    plt.plot(x[0, 0].cpu())
    plt.plot(y[0, 0].cpu())
    plt.subplot(3, 1, 2)
    plt.plot(x[0, 1].cpu())
    plt.plot(y[0, 1].cpu())
    plt.subplot(3, 1, 3)
    plt.plot(x[0, 2].cpu())
    plt.plot(y[0, 2].cpu())
    plt.show()


if __name__ == "__main__": main()
