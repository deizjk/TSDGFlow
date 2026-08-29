"""Graph divergence measures between USD-format adjacency matrices.

USD graphs are column-stochastic: A[:, source, target] is normalized along
the SOURCE axis (dim=-2), i.e. each target column is a distribution over
sources. All divergences here therefore compare per-target column
distributions along dim=-2 — NOT the last dim (a row-stochastic convention
would use dim=-1; using it here would compare the wrong objects).

The same functions are used by the zero-training diagnosis
(tools/diagnose_divergence.py) and by the x3 alignment loss, so the metric
implementation is validated once and reused verbatim.

KL direction semantics for alignment:
    column_kl(P, Q) = KL(P || Q), mass-covering w.r.t. Q.
    Alignment default (per design doc): column_kl(A_time.detach(), A_freq)
    — teacher = time graph, student = frequency graph.
"""

import torch

EPS = 1e-8


def _clamp_normalize(graph: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Make columns strictly positive and renormalized (KL-safe)."""
    g = graph.clamp_min(eps)
    return g / g.sum(dim=-2, keepdim=True)


def column_kl(p_graph: torch.Tensor, q_graph: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """KL(P || Q) per target column (dim=-2), averaged over targets.

    Args:
        p_graph, q_graph: [..., K, K] column-stochastic graphs.
    Returns:
        [...] divergence per graph (batch-shaped).
    """
    p = _clamp_normalize(p_graph, eps)
    q = _clamp_normalize(q_graph, eps)
    per_target = (p * (p.log() - q.log())).sum(dim=-2)  # [..., K_target]
    return per_target.mean(dim=-1)


def column_js(p_graph: torch.Tensor, q_graph: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Jensen-Shannon divergence per target column, averaged over targets.

    Symmetric, bounded by log(2).
    """
    p = _clamp_normalize(p_graph, eps)
    q = _clamp_normalize(q_graph, eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=-2) \
        + 0.5 * (q * (q.log() - m.log())).sum(dim=-2)
    return js.mean(dim=-1)


def graph_l1(p_graph: torch.Tensor, q_graph: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference over all edges."""
    return (p_graph - q_graph).abs().mean(dim=(-2, -1))
