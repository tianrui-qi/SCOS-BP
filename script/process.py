import torch
import numpy as np
import pandas as pd

import os
import scipy.io


def main(data_fold: str ='data/waveform/') -> None:
    # load data
    data_load_path = os.path.join(data_fold, 'raw.mat')
    data = scipy.io.loadmat(data_load_path)["data_store"][0, 0]
    # create profile
    df_sample, df_subject = createProfile(data)
    # split, update profile in-place
    split(df_sample, df_subject, name='split', ratio=(0.5, 0.2, 0.3))
    # save profile
    df_sample.to_csv(os.path.join(data_fold, 'sample.csv'), index=False)
    df_subject.to_csv(os.path.join(data_fold, 'subject.csv'), index=False)
    # save split as torch tensor
    s = torch.as_tensor(df_sample['split'].to_numpy(), dtype=torch.long)
    torch.save(s, os.path.join(data_fold, 'split.pt'))
    # save data[0] waveform and data[1] bp as torch tensors
    x = torch.from_numpy(data[0]).to(torch.float).transpose(-1, -2)
    y = torch.from_numpy(data[1]).to(torch.float)
    torch.save(x, os.path.join(data_fold, 'x.pt'))
    torch.save(y, os.path.join(data_fold, 'y.pt'))


def createProfile(data: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def split(
    df_sample: pd.DataFrame, df_subject: pd.DataFrame,  # update in-place
    name: str = 'split', 
    ratio: tuple[float, float, float] = (0.5, 0.15, 0.35), 
    seed: int = 42, iters: int = 2000,
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
        row['subject']: row['split'] for _, row in df_subject.iterrows()
    }
    df_sample[name] = df_sample['subject'].map(mapping)

    # print number of samples per split
    for s, name in enumerate(['train','valid','test']):
        total_len = int(
            df_subject.loc[df_subject['split']==s, 'length'].sum()
        )
        print(f"number of samples in {name}:\t{total_len}")


if __name__ == "__main__": main()
