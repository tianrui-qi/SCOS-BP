import hydra
import omegaconf

import torch

import matplotlib.pyplot as plt

import src


@hydra.main(
    version_base=None, config_path="../config", 
    config_name="pipeline/sanity"
)
def main(cfg: omegaconf.DictConfig) -> None:
    # device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print("using device:", device)
    # data, fixed at first batch
    dm = src.data.DataModule(**cfg.data)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x = batch[0].to(device)
    x_channel_idx = batch[1].to(device)
    # model
    model = src.model.Model(**cfg.model).to(device)
    # train
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for s in range(cfg.step):
        opt.zero_grad(set_to_none=True)
        pred, token = model.forwardReconstruction(x, x_channel_idx)
        loss = torch.nn.functional.smooth_l1_loss(pred, token)
        loss.backward()
        opt.step()
        # print current step result
        if s % 50 != 0 and s != cfg.step - 1: continue
        print(" | ".join([
            f"step={s:04d}", 
            f"loss={float(loss.detach()):.4f}",
            f"#masked={int(len(token)):4d}",
        ]))
    # eval
    model.eval()
    with torch.no_grad(): 
        # NOTE: 
        #   In our implementation, `model.forwardReconstruction` will only
        #   return reconstructed waveform for masked tokens if `user_mask`
        #   is None, i.e., default value. We will calculate loss based on
        #   these returned tokens during training. It will return 
        #   reconstructed waveform, a (B, C, T) tensor, if `user_mask` is not 
        #   None. Thus, here we set `user_mask=-1` to get the full 
        #   reconstructed waveform in order to visualize what the model 
        #   reconstructs. No other special meaning here. If you want to learn
        #   more about what `user_mask` really does, please refer to
        #   implementation and documentation of `model.forwardReconstruction`.
        x, y = model.forwardReconstruction(x, x_channel_idx, user_mask=-1)
    # plot
    plt.subplot(3, 1, 1); plt.plot(x[0, 0].cpu()); plt.plot(y[0, 0].cpu())
    plt.subplot(3, 1, 2); plt.plot(x[0, 1].cpu()); plt.plot(y[0, 1].cpu())
    plt.subplot(3, 1, 3); plt.plot(x[0, 2].cpu()); plt.plot(y[0, 2].cpu())
    plt.show()


if __name__ == "__main__": main()
