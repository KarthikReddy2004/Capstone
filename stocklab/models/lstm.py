"""Real recurrent LSTM regressor on CPU (PyTorch).

A compact, deterministic LSTM with a small MLP head, trained with Adam + MSE and
early stopping on a temporal validation tail. It is deliberately lightweight so
that a full multi-model benchmark with population-based tuning stays inside a few
minutes on a 6-core CPU with no GPU. Per-epoch train/val losses are retained for
the learning-curve chart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        head_hidden = max(8, hidden // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TorchLSTM:
    """Fit/predict wrapper around :class:`_LSTMNet`."""

    def __init__(self, params: dict, seed: int = 42):
        self.p = params
        self.seed = seed
        self.net: _LSTMNet | None = None
        self.history_ = {"train": [], "val": []}

    def fit(self, x_seq: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        n_features = x_seq.shape[2]
        hidden = int(self.p.get("n_hidden", 64))
        layers = int(self.p.get("n_layers", 1))
        dropout = float(self.p.get("dropout", 0.1))
        lr = float(self.p.get("lr", 1e-3))
        wd = float(self.p.get("alpha", 1e-4))
        batch = int(self.p.get("batch_size", 32))
        epochs = int(self.p.get("epochs", 60))
        patience = int(self.p.get("patience", 10))

        x = torch.as_tensor(x_seq, dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(y, float), dtype=torch.float32)

        # Temporal validation tail for early stopping (no shuffling across it).
        n = len(x)
        n_val = max(8, int(round(n * 0.15)))
        n_tr = max(8, n - n_val)
        x_tr, y_tr = x[:n_tr], yt[:n_tr]
        x_va, y_va = x[n_tr:], yt[n_tr:]
        if len(x_va) == 0:
            x_va, y_va = x_tr[-8:], y_tr[-8:]

        self.net = _LSTMNet(n_features, hidden, layers, dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        batch = min(max(8, batch), len(x_tr))

        best_val = float("inf")
        best_state = None
        stale = 0
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(len(x_tr), generator=g)
            epoch_loss = 0.0
            for s in range(0, len(x_tr), batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                pred = self.net(x_tr[idx])
                loss = loss_fn(pred, y_tr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 3.0)
                opt.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= max(1, len(x_tr))

            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(x_va), y_va))
            self.history_["train"].append(epoch_loss)
            self.history_["val"].append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if self.net is None or len(x_seq) == 0:
            return np.zeros(len(x_seq))
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(x_seq, dtype=torch.float32)
            return self.net(x).cpu().numpy().astype(float)


def predict_ensemble(models, x_seq: np.ndarray) -> np.ndarray:
    if len(x_seq) == 0:
        return np.zeros(0)
    preds = np.column_stack([m.predict(x_seq) for m in models])
    return preds.mean(axis=1)
