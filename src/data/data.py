import torch
import lightning
import numpy as np
import pandas as pd

import scipy.io
from typing import Literal, Final


class DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        # data
        data_load_path: str, 
        y_as_y: bool = True, 
        y_as_x: bool = False,
        # normalize
        mu: float | None = 115.0, 
        sd: float | None = 20.0,
        # split
        split_type: Literal[
            "SubjectDependent", "SubjectIndependent"
        ] | None = None,
        split_ratio: tuple[float, float, float] = (0.5, 0.2, 0.3),
        # filter
        filter_level: Literal["X|Y", "X", "Y", "X&Y", "All"] | None = None,
        **kwargs
    ) -> None:
        super().__init__()
        self._data_load_path: Final = data_load_path
        self._y_as_y: Final = y_as_y
        self._y_as_x: Final = y_as_x
        self._mu: Final = mu
        self._sd: Final = sd
        self._split_type: Final = split_type
        self._split_ratio: Final = split_ratio
        self._filter_level: Final = filter_level

        # data
        self.data_load_path: str
        self.y_as_y: bool
        self.y_as_x: bool
        self.x: torch.Tensor        # (N, C, T), float32
        self.y: torch.Tensor        # (N, ...), float32
        self.profile: pd.DataFrame  # (N, ...)
        # normalize
        self.mu: float | dict[str, float]
        self.sd: float | dict[str, float]
        # split
        self.split_type: Literal[
            "SubjectDependent", "SubjectIndependent"
        ] | None
        self.split_ratio: tuple[float, float, float] | None
        # filter
        self.filter_level: Literal["X|Y", "X", "Y", "X&Y", "All"] | None

    def setup(self, stage: str | None = None) -> None:
        # load
        self.load_(self._data_load_path, self._y_as_y, self._y_as_x)
        # normalize
        if self._y_as_y: self.normalize_(self._mu, self._sd)
        # split (https://arxiv.org/pdf/2410.03057)
        if self._split_type is not None:
            fn = getattr(self, f"split{self._split_type}_", None)
            if fn is None: raise ValueError
            fn(self._split_ratio)
        # filter
        # must be called after split and normalize to ensure consistency
        # for a given dataset
        # filter_ can only be called once
        self.filter_(self._filter_level)

    def load_(
        self, data_load_path: str, y_as_y: bool = True, y_as_x: bool = False
    ) -> None:
        # load .mat
        data = scipy.io.loadmat(data_load_path)["data_store"][0, 0]
        # x, (N, C, T)
        self.x = torch.tensor(data[0]).to(torch.float).transpose(-1, -2)
        # y, (N, ...)
        if y_as_y: 
            self.y = torch.tensor(data[1]).to(torch.float)
        # append y to last channel of x
        if y_as_y and y_as_x: 
            self.x = torch.cat([self.x, self.y.unsqueeze(1)], dim=1)
        # profile, metadata of each sample
        def scalarize(arr, dtype=None):
            out = [
                np.asarray(x).ravel()[0] 
                if np.asarray(x).size else np.nan for x in arr
            ]
            if dtype is None: return np.array(out)
            else: return np.array(out, dtype=dtype)
        self.profile = pd.DataFrame({
            'id'        : scalarize(data[2], dtype=str).flatten(),
            'group'     : scalarize(data[3], dtype=str).flatten(),
            'repeat'    : scalarize(data[4], dtype=bool).flatten(),
            'condition' : scalarize(data[5], dtype=int).flatten()
        })
        self.profile = self.profile.rename(columns={"id": "subject"})
        self.profile["health"] = self.profile["group"] != "hypertensive"
        self.profile["system"] = self.profile["group"] != "original"
        self.profile = self.profile[
            ['subject', 'health', 'system', 'repeat', 'condition']
        ]
        # update data to current setting
        self.data_load_path = data_load_path
        self.y_as_y = y_as_y
        self.y_as_x = y_as_x
        # update normalize to reflect current setting
        # we load new y, so there is no normalization yet
        if y_as_y:
            self.mu = 0.0
            self.sd = 1.0
        # update split to reflect current setting
        # we load new profile, so there is no split yet
        self.split_type = None
        self.split_ratio = None 
        # update filter to reflect current setting
        # we load new dataset, so there is no filter yet
        self.filter_level = None

    def normalize_(
        self, mu: float | None = 115.0, sd: float | None = 20.0
    ) -> None:
        # if we never load y, do nothing since normolization logic only
        # apply to y in current design
        if not self.y_as_y: return

        # in case already normalized
        self.denormalize_()     

        mu_dict = {}
        sd_dict = {}
        for s, profile_s in self.profile.groupby('subject'):
            idx_s = profile_s.index.to_numpy()
            # a subject's mu and sd
            mu_s = torch.nanmean(self.y[idx_s]).item()
            mu2_s = torch.nanmean(self.y[idx_s] * self.y[idx_s])
            sd_s = torch.sqrt(mu2_s - mu_s * mu_s).item()
            # use global mu or sd instead if given
            if mu is not None: mu_s = mu
            if sd is not None: sd_s = sd
            # apply
            self.y[idx_s] = (self.y[idx_s] - mu_s) / sd_s
            # store
            mu_dict[s] = mu_s
            sd_dict[s] = sd_s
        if self.y_as_x: self.x[:, -1, :] = self.y
        
        # update normalize paras
        self.mu = mu_dict if mu is None else mu
        self.sd = sd_dict if sd is None else sd

    def denormalize_(self) -> None:
        # if we never load y, do nothing since normolization logic only
        # apply to y in current design
        if not self.y_as_y: return

        self.y = self.denormalize()
        if self.y_as_x: self.x[:, -1, :] = self.y

        # update normalize paras
        self.mu = 0.0
        self.sd = 1.0

    def denormalize(self, y: torch.Tensor | None = None) -> torch.Tensor:
        # if we never use y, raise error since normolization logic only
        # apply to y in current design
        # paras for normalization never init if y_as_y is False and it is 
        # impossible to denormalize
        if not self.y_as_y: raise ValueError

        if y is None: y = self.y
        # sd
        if isinstance(self.sd, float):
            y = y * self.sd
        else:
            for s, profile_s in self.profile.groupby('subject'):
                idx_s = profile_s.index.to_numpy()
                y[idx_s] = y[idx_s] * self.sd[s]    # type: ignore
        # mu
        if isinstance(self.mu, float):
            y = y + self.mu
        else:
            for s, profile_s in self.profile.groupby('subject'):
                idx_s = profile_s.index.to_numpy()
                y[idx_s] = y[idx_s] + self.mu[s]    # type: ignore
        return y

    def splitSubjectDependent_(
        self,
        ratio: tuple[float, float, float] = (0.5, 0.2, 0.3),
        verbose: bool = False, seed: int = 42,
    ) -> None:
        """
        per each subject and each condition:
        -   split all samples of that condition across splits according to 
            ratio
        -   ensure each condition appears in as many splits as possible
            (ideally all three if enough samples exist)
        -   we ignore 'health', 'system', and 'repeat' here
        """
        name = self.splitSubjectDependent_.__name__
        rng = np.random.default_rng(seed)
        k = 3
        weight = np.asarray(ratio, dtype=float)
        weight = weight / weight.sum()

        # Initialize the new split column in df_sample
        df_sample = self.profile.copy()
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
                # Distribute remaining samples to splits with largest 
                # fractional parts
                frac = ideal - base
                order = np.argsort(-frac)
                base[order[:remain]] += 1
            counts = base.copy()

            # Ensure that at least min(n, k) splits receive at least one sample
            need_cover = min(n, k)
            zeros = np.where(counts == 0)[0].tolist()
            nonzero = np.where(counts > 0)[0].tolist()

            # If not enough splits are nonzero, move 1 sample from a large 
            # split to an empty one
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
        
        split = df_sample[name].to_numpy()

        # print number of samples per split
        length = tuple([int((split == s).sum()) for s in range(3)])
        if verbose: print(f"{name}: (train, valid, test) = {length}")

        self.profile["split"] = split
        self.split_type = "SubjectDependent"
        self.split_ratio = ratio

    def splitSubjectIndependent_(
        self,
        ratio: tuple[float, float, float] = (0.5, 0.2, 0.3),
        verbose: bool = False, seed: int = 42, 
        iters: int = 2000, 
    ) -> None:
        """
        assign each subject to train/valid/test (0/1/2) such that:
        -   each subject appears in exactly one split
        -   sample count per split matches the given ratio
        -   proportions of health/system/repeat True/False are similar 
            across splits
        """
        name = self.splitSubjectIndependent_.__name__

        profile_subject = pd.DataFrame({
            'subject': self.profile['subject'].unique(),
            'health': [
                self.profile[self.profile['subject'] == subject]['health']
                .iloc[0] for subject in self.profile['subject'].unique()
            ],
            'system': [
                self.profile[self.profile['subject'] == subject]['system']
                .iloc[0] for subject in self.profile['subject'].unique()
            ],
            'repeat': [
                self.profile[self.profile['subject'] == subject]['repeat']
                .iloc[0] for subject in self.profile['subject'].unique()
            ],
            'length': [
                len(self.profile[self.profile['subject'] == subject]) 
                for subject in self.profile['subject'].unique()
            ]
        })

        rng = np.random.default_rng(seed)
        subjects = profile_subject.copy()
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
        split = self.profile['subject'].map(mapping).to_numpy()

        # print number of samples per split
        length = tuple([int((split == s).sum()) for s in range(3)])
        if verbose: print(f"{name}: (train, valid, test) = {length}")

        self.profile["split"] = split
        self.split_type = "SubjectIndependent"
        self.split_ratio = ratio

    def filter_(
        self, 
        filter_level: Literal["X|Y", "X", "Y", "X&Y", "All"] | None = None
    ) -> None:
        if not self.y_as_y:
            if filter_level == "X|Y": filter_level = "X"
            # filter_level == "X": no effect
            if filter_level == "Y": filter_level = None
            if filter_level == "X&Y": filter_level = "X"
            # filter_level == "All": we will handle it below

        # raise error if already filtered
        if self.filter_level is not None: raise ValueError
        # update filter paras
        self.filter_level = filter_level

        # get valid mask according to filter_level
        if filter_level is None:  return
        elif filter_level == "X|Y":
            # at least one channel in x or y must be valid
            valid = ~(
                torch.isnan(self.x).all(dim=tuple(range(1, self.x.ndim))) &
                torch.isnan(self.y).all(dim=tuple(range(1, self.y.ndim)))
            )
        elif filter_level == "X":
            # at least one channel in x must be valid
            valid = ~torch.isnan(self.x).all(dim=tuple(range(1, self.x.ndim)))
        elif filter_level == "Y":
            # at least one channel in y must be valid
            valid = ~torch.isnan(self.y).all(dim=tuple(range(1, self.y.ndim)))
        elif filter_level == "X&Y":
            # at least one channel in x and one in y must be valid
            valid = (
                ~torch.isnan(self.x).all(dim=tuple(range(1, self.x.ndim))) &
                ~torch.isnan(self.y).all(dim=tuple(range(1, self.y.ndim)))
            )
        elif filter_level == "All" and not self.y_as_y:
            # all x channel are valid
            valid = ~(
                torch.isnan(self.x).any(dim=tuple(range(1, self.x.ndim)))
            )
        elif filter_level == "All" and self.y_as_y:
            # all x and y channel are valid
            valid = ~(
                torch.isnan(self.x).any(dim=tuple(range(1, self.x.ndim))) |
                torch.isnan(self.y).any(dim=tuple(range(1, self.y.ndim)))
            )
        else: raise ValueError

        # apply filter mask
        self.x = self.x[valid]
        self.y = self.y[valid]
        self.profile = self.profile.loc[valid.numpy()].reset_index(drop=True)

        # NOTE:
        #   this method defines only a universal NaN-based filter
        #   subclasses may extend this method to apply additional filtering 
        #   criteria


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self, 
        x: torch.Tensor,                # (N, C, T)
        y: torch.Tensor | None = None,  # (N, ...) or None
        # augment
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
    ) -> None:
        # data
        self.x = x  # (N, C, T)
        self.y = y  # (N, ...) or None
        # augment
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.channel_shift = channel_shift

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        # x
        x = self.x[i].clone()   # (C, T)
        x_channel_idx = (       # (C,)
            torch.arange(len(x), device=x.device, dtype=torch.long)
        )
        x_channel_idx[torch.all(torch.isnan(x), dim=-1)] = -1
        # augment x
        x, x_channel_idx = DataSet.augment(
            x, x_channel_idx, 
            channel_perm=self.channel_perm, 
            channel_drop=self.channel_drop, 
            channel_shift=self.channel_shift
        )
        # y
        y = self.y[i].clone() if self.y is not None else None
        # return
        return x, x_channel_idx, y

    @staticmethod
    def augment(
        x: torch.Tensor,                # (C, T)
        x_channel_idx: torch.Tensor,    # (C,)
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C, T = x.shape
        device = x.device

        # if channel_perm, randomly shuffle channels order
        if channel_perm:
            perm = torch.randperm(C, device=device)
            x, x_channel_idx = x[perm], x_channel_idx[perm]

        # if channel_drop, randomly drop some channels by setting channel
        # index to -1 and setting corresponding x to nan; 
        # keep at least one channel
        if torch.rand(()) < channel_drop:
            valid_mask = x_channel_idx != -1
            valid_idx = torch.where(valid_mask)[0]
            # randomly keep k number of channels
            k = torch.randint(
                1, len(valid_idx)+1, (1,), device=x.device
            ).item()
            perm = torch.randperm(len(valid_idx), device=x.device)
            keep_idx = valid_idx[perm][:k]
            # drop channels that valid but not keep
            drop_mask = torch.zeros_like(x_channel_idx, dtype=torch.bool)
            drop_mask[valid_idx] = True
            drop_mask[keep_idx] = False
            # update x and channel_idx
            x[drop_mask] = float("nan")
            x_channel_idx[drop_mask] = -1

        # if channel_shift in (0, 1), shift all channels by random amount
        # in [-channel_shift*T, channel_shift*T)
        # if channel_shift >= 1, shift by random amount in 
        # [-channel_shift, channel_shift]
        # pad nan for the shifted positions
        if channel_shift > 0:
            if channel_shift < 1:
                max_shift = int(T * channel_shift)
            else:
                max_shift = int(channel_shift)
            s = torch.randint(
                -max_shift, max_shift + 1, (1,), device=x.device
            ).item()
            nan = torch.full_like(x, float('nan'))
            if s > 0: nan[..., s:] = x[..., :-s]
            if s < 0: nan[..., :s] = x[..., -s:]
            if s != 0: x = nan

        return x, x_channel_idx
