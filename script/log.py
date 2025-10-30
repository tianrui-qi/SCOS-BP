import torch.utils.tensorboard
from tensorboard.backend.event_processing import event_accumulator

import os
import glob
import argparse


def main() -> None:
    args = getArgs()
    for log_fold in args.log_fold: 
        log(log_fold)


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_fold", '-L', type=str, nargs="+", required=True,
        dest="log_fold",
    )
    args = parser.parse_args()
    return args


def log(log_fold: str) -> None:
    # if train or valid subfolder exists, skip
    if os.path.exists(os.path.join(log_fold, "train")) or \
    os.path.exists(os.path.join(log_fold, "valid")):
        print(
            f"Subfolder 'train' or 'valid' already exist in {log_fold}."
            "Skipping tensorboard log rewriting."
        )
        return
    # write
    writeEvent(
        log_fold, os.path.join(log_fold, "train"), 
        rename_rules = {
            "loss/reconstruction/train" : "loss/reconstruction",
            "loss/contrastive/train"    : "loss/contrastive",
            "loss/regression/train"     : "loss/regression",
            "loss/train"                : "loss",
        }
    )
    writeEvent(
        log_fold, os.path.join(log_fold, "valid"), 
        rename_rules = {
            "loss/reconstruction/valid" : "loss/reconstruction",
            "loss/contrastive/valid"    : "loss/contrastive",
            "loss/regression/valid"     : "loss/regression",
            "loss/valid"                : "loss",
        }
    )
    writeEvent(
        log_fold, log_fold, 
        rename_rules = {
            "epoch"  : "lightning/epoch",
            "lr-Adam": "lightning/lr-Adam",
        }
    )
    # remove
    os.remove(findEvent(log_fold))
    if os.path.exists(os.path.join(log_fold, "hparams.yaml")):
        os.remove(os.path.join(log_fold, "hparams.yaml"))


def writeEvent(
    log_load_fold: str, log_save_fold: str,
    rename_rules: dict[str, str]
) -> None:
    # load old log
    log_load_path = findEvent(log_load_fold)
    ea = event_accumulator.EventAccumulator(log_load_path)
    ea.Reload()
    # save new log
    os.makedirs(log_save_fold, exist_ok=True)
    writer = torch.utils.tensorboard.SummaryWriter(     # type: ignore
        log_save_fold
    )
    for tag in ea.Tags().get("scalars", []):
        if tag not in rename_rules: continue
        new_tag = rename_rules.get(tag, tag)
        for scalar_event in ea.Scalars(tag):
            writer.add_scalar(
                new_tag,
                scalar_event.value,
                global_step=scalar_event.step,
                walltime=scalar_event.wall_time,
            )
    writer.close()


def findEvent(log_load_fold: str) -> str:
    candidates = glob.glob(
        os.path.join(log_load_fold, "events.out.tfevents.*")
    )
    if not candidates: raise FileNotFoundError(
        f"No TensorBoard event file found in: {log_load_fold}"
    )
    # find the oldest one
    candidates.sort(key=os.path.getmtime, reverse=False)
    return candidates[0]


if __name__ == "__main__": main()
