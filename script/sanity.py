import torch
import lightning

import json
import argparse
import matplotlib.pyplot as plt

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4000)
    args = ap.parse_args()

    # config
    with open("config/pretrain.json", "r") as f: config = json.load(f)
    config["data"]["batch_size"] = 16  # small batch for debugging
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data, fixed at first batch
    dm = src.data.DataModule(**config["data"])
    dm.setup()
    x, ch, _ = next(iter(dm.train_dataloader()))
    x = x.to(device)
    ch = ch.to(device)
    # model
    model = src.model.SCOST(**config["model"]).to(device)

    # train
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config["runner"]["lr"])
    for step in range(args.step):
        opt.zero_grad(set_to_none=True)

        pred, token = model(x, ch, mask=True, task="reconstruction")
        loss = torch.nn.functional.mse_loss(pred, token)

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
        x, y = model(x, ch, mask=False, task="reconstruction")
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