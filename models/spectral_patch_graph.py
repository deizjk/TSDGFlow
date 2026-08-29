import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _negative_large(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def normalize_usd_graph(graph: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize graph for USD GNN: A[:, source, target] sums over source."""
    graph = torch.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph = graph.clamp_min(0.0)
    denom = graph.sum(dim=-2, keepdim=True).clamp_min(eps)
    return graph / denom


class FrequencyPatcher(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        representation: str = "real_imag",
        drop_dc: bool = False,
        center_before_fft: bool = False,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"freq_patch_size must be positive, got {patch_size}")
        if patch_stride <= 0:
            raise ValueError(f"freq_patch_stride must be positive, got {patch_stride}")
        if representation not in {"real_imag", "magnitude", "log_magnitude"}:
            raise ValueError(
                "spectral_representation must be one of "
                f"real_imag/magnitude/log_magnitude, got {representation!r}"
            )
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.representation = representation
        self.drop_dc = drop_dc
        self.center_before_fft = center_before_fft

    def forward(self, x: torch.Tensor, return_energy: bool = False):
        """
        Args:
            x: [B, K, T] or [B, K, T, D].
            return_energy: if True, also return per-patch log spectral energy [B, P].

        Returns:
            patch_feature: [B, P, K, C_patch], or (patch_feature, log_energy) if return_energy.
        """
        if x.dim() == 3:
            # [B, K, T] -> [B, K, D=1, T]
            signal = x.unsqueeze(2)
        elif x.dim() == 4:
            # [B, K, T, D] -> [B, K, D, T]
            signal = x.permute(0, 1, 3, 2)
        else:
            raise ValueError(f"Expected input with shape [B,K,T] or [B,K,T,D], got {tuple(x.shape)}")

        if signal.size(0) <= 0 or signal.size(1) <= 0 or signal.size(2) <= 0 or signal.size(3) <= 0:
            raise ValueError(f"All input dimensions must be positive, got {tuple(x.shape)}")

        out_dtype = signal.dtype if signal.dtype.is_floating_point else torch.float32
        fft_input = signal
        if fft_input.dtype in (torch.float16, torch.bfloat16) or not fft_input.dtype.is_floating_point:
            fft_input = fft_input.float()
        if self.center_before_fft:
            fft_input = fft_input - fft_input.mean(dim=-1, keepdim=True)

        # [B, K, D, T] -> [B, K, D, F]
        spectrum = torch.fft.rfft(fft_input, dim=-1, norm="ortho")
        if self.drop_dc:
            spectrum = spectrum[..., 1:]

        log_energy = None
        if return_energy:
            log_energy = self._patch_log_energy(spectrum).to(dtype=out_dtype)

        if self.representation == "real_imag":
            patches = self._real_imag_patches(spectrum)
        else:
            values = spectrum.abs()
            if self.representation == "log_magnitude":
                values = torch.log1p(values)
            patches = self._real_patches(values)

        patches = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        patches = patches.to(dtype=out_dtype)
        if return_energy:
            return patches, log_energy
        return patches

    def _patch_log_energy(self, spectrum: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Per-patch log spectral energy on the same patch grid as the representation.

        The energy view always excludes DC (regardless of drop_dc) so that the
        1/f-dominated zeroth bin cannot monopolize the gating.
        """
        # [B, K, D, F'] with F' already excluding DC when drop_dc is True
        power = spectrum.real ** 2 + spectrum.imag ** 2
        if not self.drop_dc:
            # zero out the DC bin functionally (no in-place op on autograd tensors)
            power = F.pad(power[..., 1:], (1, 0))
        power = self._pad_frequency_axis(power)
        # [B, K, D, F_pad] -> [B, K, D, P, S]
        power_patches = power.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        if power_patches.size(-2) <= 0:
            raise RuntimeError("Frequency patching produced no patches")
        # sum over channels, feature dim and in-patch bins -> [B, P]
        energy = power_patches.sum(dim=(1, 2, 4))
        log_energy = torch.log(energy + eps)
        return torch.nan_to_num(log_energy, nan=0.0, posinf=0.0, neginf=0.0)

    def _pad_frequency_axis(self, values: torch.Tensor) -> torch.Tensor:
        freq_bins = values.size(-1)
        if freq_bins < self.patch_size:
            pad_len = self.patch_size - freq_bins
        else:
            remainder = (freq_bins - self.patch_size) % self.patch_stride
            pad_len = 0 if remainder == 0 else self.patch_stride - remainder
        if pad_len > 0:
            values = F.pad(values, (0, pad_len))
        return values

    def _real_patches(self, values: torch.Tensor) -> torch.Tensor:
        values = self._pad_frequency_axis(values)
        # [B, K, D, F_pad] -> [B, K, D, P, S]
        patches = values.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        if patches.size(-2) <= 0:
            raise RuntimeError("Frequency patching produced no patches")
        # [B, K, D, P, S] -> [B, P, K, D*S]
        patches = patches.permute(0, 3, 1, 2, 4).contiguous()
        return patches.flatten(start_dim=-2)

    def _real_imag_patches(self, spectrum: torch.Tensor) -> torch.Tensor:
        real = self._pad_frequency_axis(spectrum.real)
        imag = self._pad_frequency_axis(spectrum.imag)
        # [B, K, D, F_pad] -> [B, K, D, P, S]
        real_patches = real.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        imag_patches = imag.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        if real_patches.size(-2) <= 0:
            raise RuntimeError("Frequency patching produced no patches")
        # [B, K, D, P, S] + [B, K, D, P, S] -> [B, P, K, D, 2*S]
        patches = torch.cat([real_patches, imag_patches], dim=-1)
        patches = patches.permute(0, 3, 1, 2, 4).contiguous()
        return patches.flatten(start_dim=-2)


class FrequencyPatchEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if embed_dim <= 0:
            raise ValueError(f"freq_embed_dim must be positive, got {embed_dim}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, P, K, C_patch] -> [B, P, K, D_freq]
        return self.net(x)


class FrequencyPatchedGraphLearner(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        embed_dim: int,
        representation: str = "real_imag",
        topk: int = 0,
        dropout: float = 0.1,
        drop_dc: bool = False,
        center_before_fft: bool = False,
        input_size: int = 1,
        gating_mode: str = "mlp",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if topk < 0:
            raise ValueError(f"spectral_graph_topk must be >= 0, got {topk}")
        if gating_mode not in {"uniform", "energy", "mlp", "hybrid"}:
            raise ValueError(
                "spectral_gating_mode must be one of uniform/energy/mlp/hybrid, "
                f"got {gating_mode!r}"
            )

        multiplier = 2 if representation == "real_imag" else 1
        input_dim = patch_size * input_size * multiplier
        self.patcher = FrequencyPatcher(
            patch_size=patch_size,
            patch_stride=patch_stride,
            representation=representation,
            drop_dc=drop_dc,
            center_before_fft=center_before_fft,
        )
        self.encoder = FrequencyPatchEncoder(input_dim=input_dim, embed_dim=embed_dim, dropout=dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.patch_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.topk = topk
        self.eps = eps
        self.gating_mode = gating_mode
        if gating_mode == "hybrid":
            # Learnable energy temperature: omega_p ∝ E_p^beta * exp(MLP_p).
            self.energy_beta = nn.Parameter(torch.tensor(1.0))
            # Warm start: zero the scorer head so training begins at pure
            # energy weights (beta=1) and the MLP learns a residual correction.
            nn.init.zeros_(self.patch_scorer[-1].weight)
            nn.init.zeros_(self.patch_scorer[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, K, T] or [B, K, T, D].

        Returns:
            patch_embeddings: [B, P, K, D_freq]
            patch_graphs: [B, P, K, K] in USD graph format
            patch_weights: [B, P]
            frequency_graph: [B, K, K] in USD graph format
            frequency_descriptor: [B, D_freq]
        """
        if self.gating_mode in ("energy", "hybrid"):
            patch_features, log_energy = self.patcher(x, return_energy=True)
        else:
            patch_features = self.patcher(x)
            log_energy = None
        if patch_features.size(-1) != self.input_dim:
            raise ValueError(
                "Frequency patch feature dimension mismatch: "
                f"expected {self.input_dim}, got {patch_features.size(-1)}. "
                "Check input_size against the last feature dimension of x."
            )

        patch_embeddings = self.encoder(patch_features)
        q = self.q_proj(patch_embeddings)
        k = self.k_proj(patch_embeddings)

        # [B, P, K, D] x [B, P, D, K] -> [B, P, K, K]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        patch_graphs = self._softmax_usd(scores)

        patch_descriptor = patch_embeddings.mean(dim=2)  # [B, P, D_freq]
        if self.gating_mode == "uniform":
            patch_logits = patch_descriptor.new_zeros(patch_descriptor.shape[:2])
        elif self.gating_mode == "energy":
            patch_logits = log_energy  # beta ≡ 1, no MLP term
        elif self.gating_mode == "hybrid":
            patch_logits = (
                self.patch_scorer(patch_descriptor).squeeze(-1)
                + self.energy_beta * log_energy
            )
        else:  # "mlp": original behaviour
            patch_logits = self.patch_scorer(patch_descriptor).squeeze(-1)  # [B, P]
        patch_weights = torch.softmax(patch_logits, dim=-1)

        frequency_graph = (patch_graphs * patch_weights[:, :, None, None]).sum(dim=1)
        frequency_graph = normalize_usd_graph(frequency_graph, eps=self.eps)
        frequency_descriptor = (patch_descriptor * patch_weights[:, :, None]).sum(dim=1)

        return {
            "patch_embeddings": patch_embeddings,
            "patch_graphs": patch_graphs,
            "patch_weights": patch_weights,
            "frequency_graph": frequency_graph,
            "frequency_descriptor": frequency_descriptor,
        }

    def _softmax_usd(self, scores: torch.Tensor) -> torch.Tensor:
        num_channels = scores.size(-1)
        if self.topk > 0 and self.topk < num_channels:
            _, top_indices = scores.topk(self.topk, dim=-2)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(dim=-2, index=top_indices, value=True)
            scores = scores.masked_fill(~mask, _negative_large(scores.dtype))
        graph = torch.softmax(scores, dim=-2)
        graph = torch.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_usd_graph(graph, eps=self.eps)


class TemporalSpectralGraphFusion(nn.Module):
    def __init__(
        self,
        time_dim: int,
        freq_dim: int,
        mode: str = "gated",
        graph_alpha: float = 0.7,
        learnable_alpha_init: float = 0.7,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        valid_modes = {"time_only", "freq_only", "mean", "fixed_alpha", "learnable_alpha", "gated"}
        if mode not in valid_modes:
            raise ValueError(
                "graph_fusion_mode must be time_only/freq_only/mean/fixed_alpha/learnable_alpha/gated, "
                f"got {mode!r}"
            )
        if time_dim <= 0:
            raise ValueError(f"time_dim must be positive, got {time_dim}")
        if freq_dim <= 0:
            raise ValueError(f"freq_dim must be positive, got {freq_dim}")
        if not 0.0 <= graph_alpha <= 1.0:
            raise ValueError(f"graph_alpha must be in [0, 1], got {graph_alpha}")
        if not 0.0 < learnable_alpha_init < 1.0:
            raise ValueError(
                "learnable_alpha_init must be strictly between 0 and 1, "
                f"got {learnable_alpha_init}"
            )
        hidden_dim = max(4, (time_dim + freq_dim) // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(time_dim + freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mode = mode
        self.graph_alpha = float(graph_alpha)
        self.learnable_alpha_init = float(learnable_alpha_init)
        if self.mode == "learnable_alpha":
            alpha_logit = math.log(self.learnable_alpha_init / (1.0 - self.learnable_alpha_init))
            self.alpha_param = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        self.eps = eps

    def current_alpha(self) -> torch.Tensor:
        """Return the scalar time-graph weight for learnable-alpha fusion."""
        if self.mode != "learnable_alpha":
            raise RuntimeError("current_alpha is only available in learnable_alpha mode")
        return torch.sigmoid(self.alpha_param)

    def forward(
        self,
        time_graph: torch.Tensor,
        frequency_graph: torch.Tensor,
        time_descriptor: torch.Tensor,
        frequency_descriptor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            time_graph: [B, K, K] in USD graph format.
            frequency_graph: [B, K, K] in USD graph format.
            time_descriptor: [B, D_time].
            frequency_descriptor: [B, D_freq].
        """
        if time_graph.shape != frequency_graph.shape:
            raise ValueError(
                "time_graph and frequency_graph must have identical shapes, got "
                f"{tuple(time_graph.shape)} and {tuple(frequency_graph.shape)}"
            )
        if time_graph.dim() != 3:
            raise ValueError(f"Expected graph shape [B,K,K], got {tuple(time_graph.shape)}")

        batch_size = time_graph.size(0)
        if self.mode == "time_only":
            gate = time_graph.new_ones(batch_size, 1, 1)
            return {"fused_graph": time_graph, "gate": gate}
        if self.mode == "freq_only":
            gate = time_graph.new_zeros(batch_size, 1, 1)
            return {"fused_graph": frequency_graph, "gate": gate}
        if self.mode == "mean":
            gate = time_graph.new_full((batch_size, 1, 1), 0.5)
            fused = 0.5 * time_graph + 0.5 * frequency_graph
            return {"fused_graph": normalize_usd_graph(fused, eps=self.eps), "gate": gate}
        if self.mode == "fixed_alpha":
            gate = time_graph.new_full((batch_size, 1, 1), self.graph_alpha)
            fused = self.graph_alpha * time_graph + (1.0 - self.graph_alpha) * frequency_graph
            return {"fused_graph": normalize_usd_graph(fused, eps=self.eps), "gate": gate}
        if self.mode == "learnable_alpha":
            alpha = self.current_alpha().to(dtype=time_graph.dtype)
            gate = alpha.expand(batch_size, 1, 1)
            fused = alpha * time_graph + (1.0 - alpha) * frequency_graph
            return {"fused_graph": normalize_usd_graph(fused, eps=self.eps), "gate": gate}

        if time_descriptor.dim() != 2 or frequency_descriptor.dim() != 2:
            raise ValueError(
                "time_descriptor and frequency_descriptor must be [B,D], got "
                f"{tuple(time_descriptor.shape)} and {tuple(frequency_descriptor.shape)}"
            )
        fusion_descriptor = torch.cat([time_descriptor, frequency_descriptor], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(fusion_descriptor)).view(batch_size, 1, 1)
        fused = gate * time_graph + (1.0 - gate) * frequency_graph
        return {"fused_graph": normalize_usd_graph(fused, eps=self.eps), "gate": gate}
