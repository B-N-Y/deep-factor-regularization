"""
training.py
-----------
Training loop. No learning-rate schedule: every model is trained for a fixed
number of epochs with Adam. Two alternative regularizers can be switched on for the
comparison regimes (both off by default, which leaves every other regime unchanged):
weight decay (an L2 penalty through Adam's weight_decay) and early stopping on a
validation set (the best epoch is restored).
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import ALT_REGULARIZERS, BATCH_SIZE, GRADIENT_CLIP, LEARNING_RATE
from .models import DeepFactorModel


class NonFiniteLoss(RuntimeError):
    """Training broke down numerically (NaN/inf loss or gradient)."""


def is_out_of_memory(err: BaseException) -> bool:
    """True for GPU or host out-of-memory errors (the trial is skipped, the job goes on)."""
    if isinstance(err, MemoryError):
        return True
    oom = getattr(torch.cuda, 'OutOfMemoryError', ())
    return (oom and isinstance(err, oom)) or (isinstance(err, RuntimeError) and 'out of memory' in str(err).lower())


def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_model(arch: str, params: dict, dropout_type: str, divergence_method: str,
                n_inputs: int, device: str) -> DeepFactorModel:
    # the alternative regularizers use the network without dropout
    model = DeepFactorModel(
        arch=arch,
        n_inputs=n_inputs,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout_type='none' if dropout_type in ALT_REGULARIZERS else dropout_type,
        base_rate=params.get('base_rate', 0.0),
        alpha=params.get('alpha', 0.0),
        divergence_method=divergence_method,
    )
    return model.to(device)


def train_model(model: DeepFactorModel, X: np.ndarray, y: np.ndarray, epochs: int, device: str,
                monitor: dict = None, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
                clip_value: float = GRADIENT_CLIP, weight_decay: float = 0.0, early_stop: dict = None) -> dict:
    """
    Fit the model on (X, y) (scaled). If monitor = {'X', 'y_s'} is given, its MSE
    (scaled units, deterministic forward pass) is logged per epoch for plotting
    only; it is never used to stop or select.

    Returns per-epoch history: loss (MSE + penalty), mse, div, rate (mean dropout
    rate over sites and mini-batches) and, if monitored, monitor_mse.

    Raises NonFiniteLoss before any parameter update that would use a NaN/inf
    loss or gradient; the checks only read values.

    weight_decay > 0 adds an L2 penalty through Adam. early_stop = {'X', 'y_s', 'patience'}
    evaluates the MSE on that set after every epoch, stops after `patience` epochs without
    improvement and restores the best epoch (history['best_epoch'], counted from 1).
    """
    if weight_decay > 0:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = {'loss': [], 'mse': [], 'div': [], 'rate': []}
    if monitor is not None:
        history['monitor_mse'] = []
        mon_x = torch.tensor(monitor['X'], dtype=torch.float32).to(device)
        mon_y = torch.tensor(monitor['y_s'], dtype=torch.float32).unsqueeze(1).to(device)

    for ep in range(epochs):
        model.train()
        losses, mses, divs, rates = [], [], [], []
        for xb, tar in loader:
            xb, tar = xb.to(device), tar.to(device)
            optimizer.zero_grad()
            out, div = model(xb)
            loss = model.compute_loss(out, tar, div)
            if not torch.isfinite(loss):
                raise NonFiniteLoss(f"non-finite loss in epoch {ep + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            if not torch.isfinite(grad_norm):
                raise NonFiniteLoss(f"non-finite gradient in epoch {ep + 1}")
            optimizer.step()

            rates.append(float(np.mean(model.current_rates())))
            div_mean = div.mean().item()
            model.update_dynamic_rate(div_mean)      # takes effect from the next mini-batch
            losses.append(loss.item())
            mses.append(F.mse_loss(out, tar).item())
            divs.append(div_mean)

        history['loss'].append(float(np.mean(losses)))
        history['mse'].append(float(np.mean(mses)))
        history['div'].append(float(np.mean(divs)))
        history['rate'].append(float(np.mean(rates)))

        if monitor is not None:
            model.eval()
            with torch.no_grad():
                out_m, _ = model(mon_x)
                history['monitor_mse'].append(F.mse_loss(out_m, mon_y).item())

        if early_stop is not None:
            if ep == 0:
                es_x = torch.tensor(early_stop['X'], dtype=torch.float32).to(device)
                es_y = torch.tensor(early_stop['y_s'], dtype=torch.float32).unsqueeze(1).to(device)
                best_mse, best_ep, best_state = float('inf'), 0, None
            model.eval()
            with torch.no_grad():
                out_e, _ = model(es_x)
                mse_e = F.mse_loss(out_e, es_y).item()
            if mse_e < best_mse:
                best_mse, best_ep, best_state = mse_e, ep + 1, copy.deepcopy(model.state_dict())
            elif ep + 1 - best_ep >= early_stop['patience']:
                break

    if early_stop is not None and best_state is not None:
        model.load_state_dict(best_state)
        history['best_epoch'] = best_ep
    return history
