#!/usr/bin/env python
"""
divergence_scale.py
-------------------
Numerical illustration of Proposition 1: size, gradient and local order of the 14
consistency penalties for a representative final hidden state under dropout.

    h = tanh(z), z ~ N(0, 1)^K;   h_tilde = h * m / (1 - p),  m_k ~ Bernoulli(1 - p)
    penalty D(softmax(h), softmax(h_tilde)) with the implementations of src/divergences.py

For each divergence: the mean penalty at dropout rates 0.1 and 0.3 (K = 256), the mean
norm of its gradient with respect to h at p = 0.1, and the exponent of the penalty in
p, the slope of log mean D on log p over p in [0.01, 0.1]. Since the perturbation norm
grows like sqrt(p), a second-order divergence should have an exponent near 1 and a
first-order one near 1/2. Printed as JSON for make_tables.py.

    python 3_analysis/divergence_scale.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '1_commodity_futures'))
from src.divergences import DIVERGENCE_METHODS, compute_divergence  # noqa: E402

N, K, SEED = 2000, 256, 0
P_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]


def penalty(p: float, method: str, grad: bool = False):
    g = torch.Generator().manual_seed(SEED)
    h = torch.tanh(torch.randn(N, K, generator=g, dtype=torch.float64)).requires_grad_(grad)
    m = (torch.rand(N, K, generator=g, dtype=torch.float64) > p).to(torch.float64)
    d = compute_divergence(torch.softmax(h, dim=-1), torch.softmax(h * m / (1 - p), dim=-1), method)
    if not grad:
        return float(d.mean())
    d.sum().backward()
    return float(d.mean()), float(h.grad.norm(dim=-1).mean())


def main():
    out = {}
    for method in DIVERGENCE_METHODS:
        d01, g01 = penalty(0.1, method, grad=True)
        slope = np.polyfit(np.log(P_GRID), np.log([penalty(p, method) for p in P_GRID]), 1)[0]
        out[method] = dict(d_p01=d01, d_p03=penalty(0.3, method), grad_p01=g01, exponent=float(slope))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
