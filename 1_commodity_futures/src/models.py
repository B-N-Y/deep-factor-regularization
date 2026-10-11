"""
models.py
---------
Deep factor models: LSTM, GRU and MLP with six dropout regimes, divergence
regularization and two comparison regimes without dropout.

    lstm / gru  stacked recurrent layers over the input window (factors_s, lag1_s);
                dropout sites between layers (on the layer outputs) and on the
                final hidden state
    mlp         stacked ReLU layers on the day-t inputs; dropout sites after each
                hidden layer except the last and on the final hidden state

    none        no dropout, MSE loss
    monte_carlo rate p0 at every site, also at inference (predictions average
                MC_SAMPLES passes), MSE loss
    standard    rate p0 at every site during training only, MSE loss (plain dropout)
    fixed       rate p0 at every site, MSE + alpha * D
    dynamic     one global rate p_t = clip(p0 + alpha * mean D of the previous
                mini-batch) shared by every site, MSE + alpha * D
    adaptive    each site sets its own rate from a local divergence between its
                input and a base-rate dropped copy, MSE + alpha * D
    weight_decay, early_stopping
                no dropout, MSE loss; the regularization is applied in training.py

D is the divergence between softmax(h) and softmax(dropout(h)) for the final
hidden state h, computed on the representation itself (no extra trainable layer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MAX_DROPOUT_RATE, PENALIZED_TYPES
from .divergences import compute_divergence


class AdaptiveDropout(nn.Module):
    """Dropout whose rate is set per call from a local divergence."""

    def __init__(self, base_rate: float, alpha: float, divergence_method: str):
        super().__init__()
        self.base_rate = base_rate
        self.alpha = alpha
        self.divergence_method = divergence_method
        self.last_rate = base_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probe = F.dropout(x, p=self.base_rate, training=True)
            div = compute_divergence(F.softmax(x, dim=-1), F.softmax(probe, dim=-1),
                                     self.divergence_method).mean()
            rate = float(torch.clamp(self.base_rate + self.alpha * div, 0.0, MAX_DROPOUT_RATE))
        self.last_rate = rate
        return F.dropout(x, p=rate, training=True)


class DeepFactorModel(nn.Module):
    def __init__(self, arch: str, n_inputs: int, hidden_size: int, num_layers: int,
                 dropout_type: str = 'none', base_rate: float = 0.0,
                 alpha: float = 0.0, divergence_method: str = 'none'):
        super().__init__()
        if arch not in ('lstm', 'gru', 'mlp'):
            raise ValueError(f"Unknown architecture: {arch}")
        self.arch = arch
        self.dropout_type = dropout_type
        self.base_rate = base_rate if dropout_type != 'none' else 0.0
        self.alpha = alpha if dropout_type in PENALIZED_TYPES else 0.0
        self.divergence_method = divergence_method
        self.num_layers = num_layers
        self.dynamic_rate = self.base_rate

        if arch == 'mlp':
            self.layers = nn.ModuleList([nn.Linear(n_inputs if i == 0 else hidden_size, hidden_size)
                                         for i in range(num_layers)])
        else:
            rnn = nn.LSTM if arch == 'lstm' else nn.GRU
            self.layers = nn.ModuleList([rnn(n_inputs if i == 0 else hidden_size, hidden_size, batch_first=True)
                                         for i in range(num_layers)])
        if dropout_type == 'adaptive':
            self.adaptive = nn.ModuleList([
                AdaptiveDropout(base_rate, alpha, divergence_method) for _ in range(num_layers)
            ])
        self.fc = nn.Linear(hidden_size, 1)

    # ── dropout helpers ─────────────────────────────────────────────────────
    def _drop(self, x: torch.Tensor, site: int, stochastic: bool) -> torch.Tensor:
        if not stochastic or self.dropout_type == 'none':
            return x
        if self.dropout_type in ('fixed', 'monte_carlo', 'standard'):
            return F.dropout(x, p=self.base_rate, training=True)
        if self.dropout_type == 'dynamic':
            return F.dropout(x, p=self.dynamic_rate, training=True)
        if self.dropout_type == 'adaptive':
            return self.adaptive[site](x)
        raise ValueError(f"Unknown dropout type: {self.dropout_type}")

    def current_rates(self) -> list:
        """Dropout rate at each site after the latest forward pass."""
        if self.dropout_type == 'adaptive':
            return [m.last_rate for m in self.adaptive]
        if self.dropout_type == 'dynamic':
            return [self.dynamic_rate] * self.num_layers
        return [self.base_rate] * self.num_layers

    def update_dynamic_rate(self, div_mean: float):
        if self.dropout_type == 'dynamic':
            self.dynamic_rate = min(max(self.base_rate + self.alpha * div_mean, 0.0), MAX_DROPOUT_RATE)

    # ── forward / loss ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, mc: bool = False):
        """
        x: (batch, window, n_inputs) scaled inputs; the MLP uses x[:, -1, :].
        Dropout is active in training mode, and at inference only for
        monte_carlo with mc=True. Returns (prediction (batch, 1), divergence (batch,)).
        """
        stochastic = self.training or (mc and self.dropout_type == 'monte_carlo')
        last = self.num_layers - 1

        if self.arch == 'mlp':
            h = x[:, -1, :]
            for i, layer in enumerate(self.layers):
                h = F.relu(layer(h))
                if i < last:
                    h = self._drop(h, i, stochastic)
            h_last = h
        else:
            h = x
            for i, rnn in enumerate(self.layers):
                h, _ = rnn(h)
                if i < last:
                    h = self._drop(h, i, stochastic)
            h_last = h[:, -1, :]
        h_drop = self._drop(h_last, last, stochastic)

        if self.training and self.dropout_type in PENALIZED_TYPES:
            div = compute_divergence(F.softmax(h_last, dim=-1), F.softmax(h_drop, dim=-1),
                                     self.divergence_method)
        else:
            div = torch.zeros(x.size(0), device=x.device)
        return self.fc(h_drop), div

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor,
                     div: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(preds, target)
        if self.dropout_type in PENALIZED_TYPES:
            return mse + self.alpha * div.mean()
        return mse
