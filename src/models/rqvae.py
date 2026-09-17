"""Residual Quantisation VAE for multimodal Semantic IDs.

    item embedding e_i
        │
      Encoder  →  z
        │
      Codebook 1 → q1 ,  residual r1 = z - q1
        │
      Codebook 2 → q2 ,  residual r2 = r1 - q2
        │
      Codebook 3 → q3
        │
      codes (c1, c2, c3)   and   Decoder(z_q) → ê_i

Loss::

    L = ||e_i - ê_i||²  +  β · Σ_k ||sg(r_k) - e_k||²

Dead codes are tracked and re-seeded from live encoder outputs, which is the
difference between a usable codebook and one where 80 % of the codes are unused.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(self, latent_dim: int, num_codebooks: int = 3, codebook_size: int = 256,
                 commitment: float = 0.25) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment = commitment

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim) * (1.0 / latent_dim**0.5))
            for _ in range(num_codebooks)
        ])
        # per-level usage counter used for dead-code re-seeding
        self.register_buffer("usage", torch.zeros(num_codebooks, codebook_size), persistent=False)

    # ------------------------------------------------------------------
    def quantize(self, z: torch.Tensor, update_usage: bool = True):
        """Return ``(z_q, codes, losses, residuals)``.

        ``losses`` holds the two standard VQ-VAE terms separately:

        * ``codebook``     — ``||sg(r) - e||²``  pulls codes onto the residuals;
        * ``commitment``   — ``||r - sg(e)||²``  pulls the encoder towards the codes.

        Both matter.  Training only the codebook term (a common shortcut) leaves
        the encoder free to drift and is a direct cause of dead codes and
        Semantic-ID collisions.
        """
        residual = z
        z_q = torch.zeros_like(z)
        codes = []
        codebook_loss = z.new_zeros(())
        commit_loss = z.new_zeros(())
        residuals = []
        for k, cb in enumerate(self.codebooks):
            residuals.append(residual)
            # ||r - e||^2 = ||r||^2 - 2<r,e> + ||e||^2
            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ cb.t()
                + cb.pow(2).sum(1).unsqueeze(0)
            )
            idx = dist.argmin(dim=1)
            q = cb[idx]
            codebook_loss = codebook_loss + F.mse_loss(residual.detach(), q)
            commit_loss = commit_loss + F.mse_loss(residual, q.detach())
            z_q = z_q + q
            residual = residual - q
            codes.append(idx)
            if update_usage and self.training:
                self.usage[k] += torch.bincount(idx.detach(), minlength=self.codebook_size).float()
        codes = torch.stack(codes, dim=1)  # (B, K)
        losses = {"codebook": codebook_loss, "commitment": commit_loss}
        return z_q, codes, losses, residuals

    @torch.no_grad()
    def measure_utilization(self, z: torch.Tensor) -> list[float]:
        """Fresh utilisation pass: fraction of codes actually selected."""
        used = torch.zeros(self.num_codebooks, self.codebook_size, dtype=torch.bool, device=z.device)
        residual = z
        for k, cb in enumerate(self.codebooks):
            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ cb.t()
                + cb.pow(2).sum(1).unsqueeze(0)
            )
            idx = dist.argmin(dim=1)
            used[k][idx] = True
            residual = residual - cb[idx]
        return [float(u.float().mean()) for u in used]

    @torch.no_grad()
    def init_from_data(self, z: torch.Tensor) -> None:
        """Seed the codebooks with real encoder outputs (residual-wise).

        Random initialisation leaves most codes far from any residual, so they
        are never selected and never receive gradient.  Seeding from data is the
        cheapest effective fix.
        """
        residual = z
        for k, cb in enumerate(self.codebooks):
            n = cb.shape[0]
            if z.shape[0] >= n:
                perm = torch.randperm(z.shape[0], device=z.device)[:n]
                cb.data.copy_(residual[perm] + 0.01 * torch.randn_like(residual[perm]))
            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ cb.t()
                + cb.pow(2).sum(1).unsqueeze(0)
            )
            idx = dist.argmin(dim=1)
            residual = residual - cb[idx]

    def straight_through(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        return z + (z_q - z).detach()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reseed_dead_codes(self, z_pool: torch.Tensor, threshold: float = 1.0) -> int:
        """Replace dead codes with *level-specific* residuals.

        Seeding level ``k`` with full encoder outputs (a common shortcut) makes
        those codes unreachable, because by level ``k`` the residual is much
        smaller than ``z``.  Each level is therefore reseeded from its own
        residual, recomputed after every level update.
        """
        if z_pool.shape[0] == 0:
            return 0
        n_reseeded = 0
        residual = z_pool
        for k, cb in enumerate(self.codebooks):
            dead = torch.nonzero(self.usage[k] < threshold).flatten()
            if dead.numel():
                take = min(dead.numel(), residual.shape[0])
                perm = torch.randperm(residual.shape[0], device=residual.device)[:take]
                cb.data[dead[:take]] = residual[perm] + 0.01 * torch.randn_like(residual[perm])
                n_reseeded += int(take)
            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ cb.t()
                + cb.pow(2).sum(1).unsqueeze(0)
            )
            residual = residual - cb[dist.argmin(dim=1)]
        self.reset_usage()
        return n_reseeded

    def reset_usage(self) -> None:
        self.usage.zero_()

    def utilization(self) -> list[float]:
        """Fraction of codes used at least once since the last reset."""
        return [float((self.usage[k] > 0).float().mean()) for k in range(self.num_codebooks)]


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int | None = None,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        commitment: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(latent_dim, input_dim // 2)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ResidualQuantizer(latent_dim, num_codebooks, codebook_size, commitment)
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, update_usage: bool = True):
        z = self.encoder(x)
        z_q, codes, losses, _ = self.quantizer.quantize(z, update_usage=update_usage)
        x_hat = self.decoder(self.quantizer.straight_through(z, z_q))
        recon = F.mse_loss(x_hat, x)
        loss = recon + losses["codebook"] + self.quantizer.commitment * losses["commitment"]
        return {
            "loss": loss,
            "reconstruction": recon,
            "commitment": losses["commitment"],
            "codebook": losses["codebook"],
            "codes": codes,
            "z": z,
            "x_hat": x_hat,
        }

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        """``(N, K)`` integer codes for each row of ``x``."""
        self.eval()
        out = []
        for i in range(0, x.shape[0], batch_size):
            z = self.encoder(x[i : i + batch_size])
            _, codes, _, _ = self.quantizer.quantize(z, update_usage=False)
            out.append(codes)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        self.eval()
        out = []
        for i in range(0, x.shape[0], batch_size):
            z = self.encoder(x[i : i + batch_size])
            z_q, _, _, _ = self.quantizer.quantize(z, update_usage=False)
            out.append(self.decoder(z_q))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def utilization(self, x: torch.Tensor, batch_size: int = 8192) -> list[float]:
        self.eval()
        zs = [self.encoder(x[i : i + batch_size]) for i in range(0, x.shape[0], batch_size)]
        return self.quantizer.measure_utilization(torch.cat(zs, dim=0))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
