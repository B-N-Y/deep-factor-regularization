"""
divergences.py
--------------
Information-theoretic divergence measures for comparing probability distributions.
Each function takes two probability tensors (orig_prob, drop_prob) and returns
a divergence value per sample (batch dimension).
"""
import math
import torch

EPS = 1e-10  # Numerical floor to avoid division by zero


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler divergence: D_KL(P || Q)"""
    return torch.sum(p * torch.log((p + EPS) / (q + EPS)), dim=-1)


def reverse_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Reverse KL divergence: D_KL(Q || P)"""
    return torch.sum(q * torch.log((q + EPS) / (p + EPS)), dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (symmetric)"""
    m = 0.5 * (p + q)
    kl1 = torch.sum(p * torch.log((p + EPS) / (m + EPS)), dim=-1)
    kl2 = torch.sum(q * torch.log((q + EPS) / (m + EPS)), dim=-1)
    return 0.5 * (kl1 + kl2)


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hellinger distance: H(P, Q)"""
    return torch.sqrt(torch.sum((torch.sqrt(p + EPS) - torch.sqrt(q + EPS))**2, dim=-1)) / math.sqrt(2)


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Total Variation distance"""
    return 0.5 * torch.sum(torch.abs(p - q), dim=-1)


def pearson_chi_square(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Pearson chi-squared divergence"""
    return torch.sum((p - q)**2 / (q + EPS), dim=-1)


def wasserstein_1(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Wasserstein-1 (Earth Mover's) distance for 1D distributions"""
    return torch.sum(torch.abs(torch.cumsum(p, dim=-1) - torch.cumsum(q, dim=-1)), dim=-1)


def bregman_euclidean(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Euclidean Bregman divergence (squared L2)"""
    return torch.sum((p - q)**2, dim=-1)


def bregman_exponential(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exponential Bregman divergence"""
    return torch.sum(torch.exp(p) - torch.exp(q) - (p - q) * torch.exp(q), dim=-1)


def bregman_log(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Logarithmic Bregman divergence"""
    return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)) - (p - q), dim=-1)


def renyi_divergence(p: torch.Tensor, q: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Rényi divergence of order alpha (default: 2)"""
    return torch.log(torch.sum((p**alpha) / (q + EPS)**(alpha - 1), dim=-1))


def jensen_renyi(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Rényi distance: 0.5 * (JS + Renyi_2)"""
    js_div = js_divergence(p, q)
    renyi_div = torch.log(torch.sum((p**2) / (q + EPS), dim=-1))
    return 0.5 * (js_div + renyi_div)


def bhattacharyya_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Bhattacharyya distance"""
    return -torch.log(torch.sum(torch.sqrt(p * q) + EPS, dim=-1))


def tsallis_divergence(p: torch.Tensor, q: torch.Tensor, q_param: float = 2.0) -> torch.Tensor:
    """Tsallis divergence with q=2"""
    return torch.sum(p**q_param / (q + EPS), dim=-1) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Unified Dispatch Function
# ─────────────────────────────────────────────────────────────────────────────

DIVERGENCE_METHODS = {
    'kl': kl_divergence,
    'reverse_kl': reverse_kl_divergence,
    'js': js_divergence,
    'hellinger': hellinger_distance,
    'tv': total_variation,
    'pearson_chi_square': pearson_chi_square,
    'wasserstein': wasserstein_1,
    'bregman_euclidean': bregman_euclidean,
    'bregman_exponential': bregman_exponential,
    'bregman_log': bregman_log,
    'renyi': renyi_divergence,
    'jensen_renyi': jensen_renyi,
    'bhattacharyya': bhattacharyya_distance,
    'tsallis': tsallis_divergence,
}


def compute_divergence(orig_prob: torch.Tensor, drop_prob: torch.Tensor,
                       method: str = 'js') -> torch.Tensor:
    """
    Compute divergence between two probability distributions.
    
    Args:
        orig_prob: Original probability distribution (batch_size, dim)
        drop_prob: Dropped probability distribution (batch_size, dim)
        method: Name of the divergence measure (see DIVERGENCE_METHODS)
    
    Returns:
        Divergence value per sample (batch_size,)
    """
    if method not in DIVERGENCE_METHODS:
        return torch.zeros_like(orig_prob[:, 0])
    return DIVERGENCE_METHODS[method](orig_prob, drop_prob)
