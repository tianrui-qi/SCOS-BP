import torch
import numpy as np
import pandas as pd

import os
import argparse
import scipy.io


def main() -> None:
    args = getArgs()
    process(args.data_fold)


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fold", "-D", type=str, required=True)
    args = parser.parse_args()
    return args


def process(data_fold: str) -> None:
    # load data
    data_load_path = os.path.join(data_fold, 'raw.mat')
    data = scipy.io.loadmat(data_load_path)["data_store"][0, 0]
    # filer out sample that all nan in x and y
    valid = ~(np.isnan(data[0]).all((1, 2)) & np.isnan(data[1]).all((1)))
    data = [d[valid] for d in data]
    # create profile
    df_sample, df_subject = createProfile(data)
    # split, update profile in-place
    split01(df_sample, df_subject, ratio=(0.5, 0.2, 0.3), name='split01')
    split02(df_sample, df_subject, ratio=(0.5, 0.2, 0.3), name='split02')
    split03(df_sample, df_subject, ratio=(0.5, 0.2, 0.3), name='split03')
    # save profile
    df_sample.to_csv(os.path.join(data_fold, 'sample.csv'), index=False)
    df_subject.to_csv(os.path.join(data_fold, 'subject.csv'), index=False)
    # save split as torch tensor
    s01 = torch.as_tensor(df_sample['split01'].to_numpy(), dtype=torch.long)
    torch.save(s01, os.path.join(data_fold, 'split01.pt'))
    s02 = torch.as_tensor(df_sample['split02'].to_numpy(), dtype=torch.long)
    torch.save(s02, os.path.join(data_fold, 'split02.pt'))
    s03 = torch.as_tensor(df_sample['split03'].to_numpy(), dtype=torch.long)
    torch.save(s03, os.path.join(data_fold, 'split03.pt'))
    # save data[0] waveform and data[1] bp as torch tensors
    x = torch.from_numpy(data[0]).to(torch.float).transpose(-1, -2)
    y = torch.from_numpy(data[1]).to(torch.float)
    torch.save(x, os.path.join(data_fold, 'x.pt'))
    torch.save(y, os.path.join(data_fold, 'y.pt'))


def createProfile(data: list[np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # create sample level profile
    def scalarize(arr, dtype=None):
        out = [
            np.asarray(x).ravel()[0] 
            if np.asarray(x).size else np.nan for x in arr
        ]
        if dtype is None: return np.array(out)
        else: return np.array(out, dtype=dtype)
    df_sample = pd.DataFrame({
        'id'        : scalarize(data[2], dtype=str).flatten(),
        'group'     : scalarize(data[3], dtype=str).flatten(),
        'repeat'    : scalarize(data[4], dtype=bool).flatten(),
        'condition' : scalarize(data[5], dtype=int).flatten()
    })
    df_sample = df_sample.rename(columns={"id": "subject"})
    df_sample["health"] = df_sample["group"] != "hypertensive"
    df_sample["system"] = df_sample["group"] != "original"
    df_sample = df_sample[
        ['subject', 'health', 'system', 'repeat', 'condition']
    ]
    # create subject level profile
    df_subject = pd.DataFrame({
        'subject': df_sample['subject'].unique(),
        'health': [
            df_sample[df_sample['subject'] == subject]['health'].iloc[0] 
            for subject in df_sample['subject'].unique()
        ],
        'system': [
            df_sample[df_sample['subject'] == subject]['system'].iloc[0] 
            for subject in df_sample['subject'].unique()
        ],
        'repeat': [
            df_sample[df_sample['subject'] == subject]['repeat'].iloc[0] 
            for subject in df_sample['subject'].unique()
        ],
        'length': [
            len(df_sample[df_sample['subject'] == subject]) 
            for subject in df_sample['subject'].unique()
        ]
    })
    # return
    return df_sample, df_subject


def split01(
    df_sample: pd.DataFrame, df_subject: pd.DataFrame,  # update in-place
    ratio: tuple[float, float, float] = (0.5, 0.15, 0.35), 
    name: str = 'split01', seed: int = 42, iters: int = 2000,
) -> None:
    """
    Assign each subject to train/valid/test (0/1/2) such that:
      - Each subject appears in exactly one split.
      - The total 'length' (sample count) per split matches the given ratio.
      - The proportions of health/system/repeat True/False are similar across 
        splits (weighted by 'length').
    """
    rng = np.random.default_rng(seed)
    subjects = df_subject.copy()
    subjects['group'] = list(zip(
        subjects['health'], subjects['system'], subjects['repeat']
    ))
    groups = sorted(subjects['group'].unique())
    total_weight = subjects['length'].sum()
    k = 3
    weight = np.array(ratio, dtype=float)
    weight = weight / weight.sum()
    target_total = weight * total_weight

    # overall group totals
    group_totals = subjects.groupby('group')['length'].sum().to_dict()
    # target per split per group
    target_group = {g: weight * group_totals[g] for g in groups}

    best_assign = None
    best_score = np.inf

    # precompute items
    items = subjects[['subject','group','length']].to_numpy()

    def score(split_totals, split_group_totals):
        # squared error on totals + group totals
        s = np.sum((split_totals - target_total)**2)
        for g in groups:
            s += np.sum((split_group_totals[g] - target_group[g])**2)
        return s

    for _ in range(iters):
        # random shuffle order (heavier first sometimes helps)
        order = np.arange(len(items))
        rng.shuffle(order)
        # sometimes sort by descending weight then shuffle small noise
        if rng.random() < 0.5:
            order = order[np.argsort(
                -subjects['length'].to_numpy()[order] + \
                rng.normal(0, 1e-6, size=len(order))
            )]
        split_totals = np.zeros(k, dtype=float)
        split_group_totals = {g: np.zeros(k, dtype=float) for g in groups}
        assign = np.full(len(items), -1, dtype=int)
        for idx in order:
            subj, g, w = items[idx]
            # evaluate marginal cost of placing in each split
            costs = []
            for s in range(k):
                new_totals = split_totals.copy()
                new_totals[s] += w
                new_group = {
                    gg: split_group_totals[gg].copy() for gg in groups
                }
                new_group[g][s] += w
                # marginal score
                c = (
                    np.sum((new_totals - target_total)**2) - \
                    np.sum((split_totals - target_total)**2)
                )
                c += (
                    np.sum((new_group[g] - target_group[g])**2) - \
                    np.sum((split_group_totals[g] - target_group[g])**2)
                )
                # small regularizer to keep set sizes balanced early
                c += 1e-6 * (new_totals[s])
                costs.append(c)
            s_star = int(np.argmin(costs))
            assign[idx] = s_star
            split_totals[s_star] += w
            split_group_totals[g][s_star] += w
        sc = score(split_totals, split_group_totals)
        if sc < best_score:
            best_score = sc
            best_assign = assign.copy()

    # map items index to subject->split
    idx_to_subj = {i: items[i][0] for i in range(len(items))}
    mapping = {
        idx_to_subj[i]: int(best_assign[i])     # type: ignore
        for i in range(len(best_assign))        # type: ignore
    }
    df_subject[name] = df_subject['subject'].map(mapping)

    # map back to sample level
    mapping = {
        row['subject']: row[name] for _, row in df_subject.iterrows()
    }
    df_sample[name] = df_sample['subject'].map(mapping)

    # print number of samples per split
    print(name)
    for s, n in enumerate(['train','valid','test']):
        total_len = int(
            df_subject.loc[df_subject[name]==s, 'length'].sum()
        )
        print(f"number of samples in {n}:\t{total_len}")


def split02(
    df_sample: pd.DataFrame, df_subject: pd.DataFrame,  # updated in place
    ratio: tuple[float, float, float] = (0.5, 0.15, 0.35), 
    name: str = "split02", seed: int = 42,
) -> None:
    """
    Per-subject & per-condition splitting.

    For each subject and each condition:
      - Split all samples of that condition across splits according to ratio.
      - Ensure each condition appears in as many splits as possible
        (ideally all three if enough samples exist).
      - We ignore 'health', 'system', and 'repeat' here.
    
    After splitting:
      - df_sample[name]: 0(train), 1(valid), 2(test)
      - df_subject[name]: all -1 (since every subject appears in all splits)
    """
    rng = np.random.default_rng(seed)
    k = 3
    weight = np.asarray(ratio, dtype=float)
    weight = weight / weight.sum()

    # Initialize the new split column in df_sample
    if name in df_sample.columns:
        df_sample.drop(columns=[name], inplace=True)
    df_sample[name] = -1

    # Group samples by (subject, condition)
    grouped = df_sample.groupby(["subject", "condition"], sort=False)

    for (subject, cond), idx_frame in grouped:
        idx = idx_frame.index.to_numpy()
        n = len(idx)
        if n == 0:
            continue

        # Compute ideal counts for each split based on ratio
        ideal = n * weight
        base = np.floor(ideal).astype(int)
        remain = n - base.sum()
        if remain > 0:
            # Distribute remaining samples to splits with largest fractional 
            # parts
            frac = ideal - base
            order = np.argsort(-frac)
            base[order[:remain]] += 1
        counts = base.copy()

        # Ensure that at least min(n, k) splits receive at least one sample
        need_cover = min(n, k)
        zeros = np.where(counts == 0)[0].tolist()
        nonzero = np.where(counts > 0)[0].tolist()

        # If not enough splits are nonzero, move 1 sample from a large split 
        # to an empty one
        while len(nonzero) < need_cover:
            if not zeros:
                # No zero splits left but still not enough coverage → break
                break
            target = zeros.pop(0)
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[target] += 1
            nonzero = np.where(counts > 0)[0].tolist()

        # Assign samples to splits according to counts
        rng.shuffle(idx)
        start = 0
        for split_id in range(k):
            c = int(counts[split_id])
            if c <= 0:
                continue
            sel = idx[start:start + c]
            df_sample.loc[sel, name] = split_id
            start += c

        # Fallback: if any samples are still unassigned, put them into the 
        # smallest split
        if (df_sample.loc[idx, name] < 0).any():
            leftover_mask = df_sample.loc[idx, name] < 0
            cur_counts = np.array([
                (df_sample.loc[idx, name] == s).sum() for s in range(k)
            ])
            target_split = int(np.argmin(cur_counts))
            df_sample.loc[
                idx[leftover_mask.to_numpy().nonzero()[0]], name
            ] = target_split

    # At subject level, set split02 = -1 because every subject appears in 
    # all splits
    df_subject[name] = -1

    # print number of samples per split
    print(name)
    for s, n in enumerate(['train','valid','test']):
        total_len = int(
            (df_sample[name]==s).sum()
        )
        print(f"number of samples in {n}:\t{total_len}")
    # for subject in df_sample["subject"].unique():
    #     sub_df = df_sample[df_sample["subject"] == subject]
    #     total = len(sub_df)
    #     counts = [(sub_df[name] == s).sum() for s in range(k)]
    #     print(
    #         f"Subject {subject}:\ttotal={total}\t"
    #         f"split0={counts[0]}\tsplit1={counts[1]}\tsplit2={counts[2]}"
    #     )


def split03(
    df_sample: pd.DataFrame, df_subject: pd.DataFrame,  # update in-place
    ratio: tuple[float, float, float] = (0.5, 0.15, 0.35), 
    name: str = 'split03', seed: int = 42, iters: int = 2000,
) -> None:
    # check if column split01 exists
    if 'split01' not in df_sample.columns:
        split01(df_sample, df_subject, ratio=ratio, name='split01')
    # in addition to split01, put all samples that condition==1 into train set
    df_sample[name] = df_sample['split01']
    df_sample.loc[df_sample['condition'] == 1, name] = 0
    df_subject[name] = -1
    # print number of samples per split
    print(name)
    for s, n in enumerate(['train','valid','test']):
        total_len = int(
            (df_sample[name]==s).sum()
        )
        print(f"number of samples in {n}:\t{total_len}")


if __name__ == "__main__": main()
