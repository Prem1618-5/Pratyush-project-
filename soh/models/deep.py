"""Stage 6: sequence models (LSTM, GRU, 1D-CNN) on windows of k consecutive cycles.

A window always comes from ONE cell and is built AFTER the split, so no window mixes
cells or crosses train/test. The first k-1 cycles of a cell are left-padded with the
cell's first row, so every cycle gets a prediction (fair comparison with tabular models).
Early stopping uses validation cells carved out of the 41 training cells, never test cells.
"""
import copy

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


def build_windows(df, feats, k, label="soh"):
    """Return X (n, k, F), y (n,), row index (n,) -- one window per row of df."""
    Xs, ys, idx = [], [], []
    for _, d in df.groupby("cell_id", sort=False):
        d = d.sort_values("cycle")
        a = d[feats].to_numpy(np.float32)
        pad = np.vstack([np.repeat(a[:1], k - 1, axis=0), a])
        win = np.lib.stride_tricks.sliding_window_view(pad, (k, a.shape[1]))[:, 0]
        Xs.append(win)
        ys.append(d[label].to_numpy(np.float32))
        idx.append(d.index.to_numpy())
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(idx)


class RNN(nn.Module):
    def __init__(self, n_feat, cell="lstm"):
        super().__init__()
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.r1 = R(n_feat, 64, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.r2 = R(64, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        h, _ = self.r1(x)
        h, _ = self.r2(self.drop(h))
        return self.head(h[:, -1]).squeeze(-1)          # last time step only


class CNN1D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):                                 # (B, k, F) -> (B, F, k)
        return self.net(x.transpose(1, 2)).squeeze(-1)


def make_net(name, n_feat):
    return CNN1D(n_feat) if name == "cnn" else RNN(n_feat, cell=name)


class Scaler:
    """Median-impute + MinMax, fit on training rows only."""

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.med = np.nanmedian(flat, 0)
        flat = np.where(np.isnan(flat), self.med, flat)
        self.lo, self.hi = flat.min(0), flat.max(0)
        return self

    def transform(self, X):
        X = np.where(np.isnan(X), self.med, X)
        return np.clip((X - self.lo) / np.maximum(self.hi - self.lo, 1e-9), -1, 2).astype(np.float32)


def train_net(name, Xtr, ytr, Xva, yva, seed=0, epochs=40, patience=6, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = Scaler().fit(Xtr)
    mu, sd = float(ytr.mean()), float(ytr.std())
    tx, ty = torch.tensor(scaler.transform(Xtr)), torch.tensor((ytr - mu) / sd)
    vx, vy = torch.tensor(scaler.transform(Xva)), torch.tensor((yva - mu) / sd)
    net = make_net(name, Xtr.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn, best, best_state, bad = nn.MSELoss(), np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(tx), generator=g).split(batch):
            opt.zero_grad()
            loss = loss_fn(net(tx[b]), ty[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(vx), vy))
        sched.step(v)
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return {"net": net, "scaler": scaler, "mu": mu, "sd": sd, "epochs": ep + 1,
            "val_rmse": float(np.sqrt(best) * sd)}


def predict_net(model, X, batch=4096):
    xs = torch.tensor(model["scaler"].transform(X))
    with torch.no_grad():
        out = torch.cat([model["net"](xs[i:i + batch]) for i in range(0, len(xs), batch)])
    return out.numpy() * model["sd"] + model["mu"]


def n_params(net):
    return sum(p.numel() for p in net.parameters())
