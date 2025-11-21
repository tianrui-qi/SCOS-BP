import torch
import lightning

import argparse
import dataclasses
import matplotlib.pyplot as plt

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1000)
    args = ap.parse_args()
    # config
    config = src.config.Config()
    config.eval()   # turn off all augmentations
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    # data, fixed at first batch
    dm = src.data.DataModuleRaw(**dataclasses.asdict(config.data))
    dm.setup()
    x, channel_idx = next(iter(dm.train_dataloader()))
    x, channel_idx = x.to(device), channel_idx.to(device)
    # model
    model = src.model.SCOST(**dataclasses.asdict(config.model)).to(device)
    # train
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.runner.lr)
    for step in range(args.step):
        opt.zero_grad(set_to_none=True)
        pred, token = model.forwardReconstructionRaw(x, channel_idx)
        loss = torch.nn.functional.smooth_l1_loss(pred, token)
        loss.backward()
        opt.step()
        if step % 50 != 0 and step != args.step - 1: continue
        print(" | ".join([
            f"step={step:04d}", 
            f"loss={float(loss):.6f}",
            f"#masked={int(len(token))}",
        ]))
    # eval
    model.eval()
    with torch.no_grad():
        x, y = model.forwardReconstructionRaw(x, channel_idx, user_mask=3)
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
