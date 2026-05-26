"""
EEG-LeJEPA + Region Masking: LeJEPA with brain region spatial masking.

In addition to temporal block masking, randomly masks entire brain regions
(frontal, central, parietal, temporal, occipital) during pretraining.

This forces the model to learn cross-region functional connectivity:
"If parietal is masked, can you predict it from frontal + temporal?"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock, DynamicChannelMixer
from config import BrainWMConfig


# ============================================================
# Region Masking Module
# ============================================================

class RegionMasker(nn.Module):
    """Masks entire brain regions in the token representation.

    After tokenization, each token is [D] = [R * d_per_region].
    Region masking zeros out entire region slices and replaces
    with learnable mask tokens.

    Prediction target: reconstruct masked regions from unmasked ones.
    """

    def __init__(self, n_regions: int = 5, d_per_region: int = None,
                 d_model: int = 256, region_mask_prob: float = 0.5):
        super().__init__()
        self.n_regions = n_regions
        self.d_model = d_model
        self.region_mask_prob = region_mask_prob

        # If tokenizer doesn't have explicit regions, we treat d_model
        # as divided into n_regions equal parts
        self.d_per_region = d_per_region or (d_model // n_regions)

        # Learnable mask token per region
        self.region_mask_tokens = nn.Parameter(
            torch.randn(n_regions, self.d_per_region) * 0.02
        )

        # Cross-attention predictor: predict masked regions from unmasked
        self.region_predictor = nn.Sequential(
            nn.Linear(self.d_per_region, self.d_per_region * 2),
            nn.GELU(),
            nn.Linear(self.d_per_region * 2, self.d_per_region),
        )

    def apply_mask(self, tokens: torch.Tensor):
        """Apply region masking to encoded tokens.

        Args:
            tokens: [B, N, D] encoded tokens
        Returns:
            masked_tokens: [B, N, D] with some region slices replaced
            mask_info: dict with indices and original values for loss
        """
        if not self.training or torch.rand(1).item() > self.region_mask_prob:
            return tokens, None

        B, N, D = tokens.shape
        d = self.d_per_region
        R = self.n_regions

        # Randomly mask 1-2 regions
        n_mask = torch.randint(1, 3, (1,)).item()
        perm = torch.randperm(R, device=tokens.device)
        masked_idx = perm[:n_mask]
        unmasked_idx = perm[n_mask:]

        # Replace masked region slices with mask tokens
        region_mask = torch.zeros(D, dtype=torch.bool, device=tokens.device)
        token_values = torch.zeros(D, device=tokens.device)
        for r in masked_idx:
            start, end = r.item() * d, (r.item() + 1) * d
            if end > D:
                end = D
            region_mask[start:end] = True
            token_values[start:end] = self.region_mask_tokens[r, :end-start]

        masked_tokens = torch.where(
            region_mask.unsqueeze(0).unsqueeze(0),
            token_values.unsqueeze(0).unsqueeze(0),
            tokens,
        )

        # Extract original and predicted region features for loss
        original_regions = []
        for r in masked_idx:
            start, end = r.item() * d, min((r.item() + 1) * d, D)
            original_regions.append(tokens[:, :, start:end])

        # Predict from unmasked regions
        unmasked_features = []
        for r in unmasked_idx:
            start, end = r.item() * d, min((r.item() + 1) * d, D)
            unmasked_features.append(tokens[:, :, start:end])

        # Mean of unmasked regions as context
        if unmasked_features:
            context = torch.stack(unmasked_features, dim=0).mean(dim=0)  # [B, N, d]
        else:
            context = torch.zeros(B, N, d, device=tokens.device)

        predicted_regions = self.region_predictor(context)  # [B, N, d]

        return masked_tokens, {
            "masked_idx": masked_idx,
            "original_regions": original_regions,
            "predicted_regions": predicted_regions,
        }

    def compute_region_loss(self, mask_info):
        """Compute L2 loss between predicted and original masked regions."""
        if mask_info is None:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0, device=mask_info["predicted_regions"].device)
        for orig in mask_info["original_regions"]:
            pred = mask_info["predicted_regions"]
            # Align dimensions
            d = min(pred.shape[-1], orig.shape[-1])
            total_loss = total_loss + F.mse_loss(pred[..., :d], orig[..., :d].detach())

        return total_loss / max(len(mask_info["original_regions"]), 1)


# ============================================================
# EEG-LeJEPA + Region Masking
# ============================================================

class EEGLeJEPARegion(nn.Module):
    """LeJEPA with brain region spatial masking.

    Temporal block masking (learn time dynamics) +
    Region masking (learn spatial functional connectivity).
    """

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,
        d_model: int = 256,
        d_channel: int = 32,
        n_queries: int = 16,
        n_regions: int = 5,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,
        region_mask_prob: float = 0.5,
        region_mask_weight: float = 1.0,
        sigreg_lambda: float = 0.05,
        query_spec_weight: float = 0.1,
        n_subjects: int = 109,
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_lambda = sigreg_lambda
        self.query_spec_weight = query_spec_weight
        self.region_mask_weight = region_mask_weight

        # Tokenizer
        self.tokenizer = DynamicChannelMixer(
            n_channels, state_samples, d_model, d_channel, n_queries,
        )

        # Region masker
        self.region_masker = RegionMasker(
            n_regions=n_regions,
            d_model=d_model,
            region_mask_prob=region_mask_prob,
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _tokenize(self, eeg):
        tokens = self.tokenizer(eeg)
        N = tokens.shape[1]
        return tokens + self.pos_embed[:, :N, :]

    def _encode(self, tokens):
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def _generate_block_mask(self, B, N, device):
        n_mask = int(N * self.mask_ratio)
        n_vis = N - n_mask
        bs = self.mask_block_size
        all_vis, all_mask = [], []
        for b in range(B):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            attempts = 0
            while mask.sum() < n_mask and attempts < 100:
                start = torch.randint(0, N, (1,)).item()
                length = torch.randint(max(1, bs-2), bs+3, (1,)).item()
                mask[start:min(start+length, N)] = True
                attempts += 1
            if mask.sum() > n_mask:
                pos = mask.nonzero(as_tuple=True)[0]
                mask[pos[torch.randperm(len(pos))[:mask.sum()-n_mask]]] = False
            elif mask.sum() < n_mask:
                unm = (~mask).nonzero(as_tuple=True)[0]
                mask[unm[torch.randperm(len(unm))[:n_mask-mask.sum()]]] = True
            all_vis.append((~mask).nonzero(as_tuple=True)[0])
            all_mask.append(mask.nonzero(as_tuple=True)[0])
        return torch.stack(all_vis), torch.stack(all_mask), n_vis, n_mask

    def forward(self, eeg, return_predictions=True):
        B, T, C = eeg.shape
        N = T // self.state_samples

        all_tokens = self._tokenize(eeg)

        # Apply region masking BEFORE encoder (encoder sees spatially incomplete tokens)
        region_mask_info = None
        if self.training and return_predictions:
            all_tokens, region_mask_info = self.region_masker.apply_mask(all_tokens)

        all_encoded = self._encode(all_tokens)

        if not return_predictions:
            return {"brain_states": all_encoded}

        # Temporal block masking
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(B, N, eeg.device)

        vis_encoded = torch.gather(all_encoded, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model))
        mask_encoded = torch.gather(all_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model))

        vis_context = vis_encoded.mean(dim=1, keepdim=True).expand(-1, n_mask, -1)
        predictions = self.pred_head(vis_context)

        return {
            "predictions": predictions,
            "targets": mask_encoded,
            "all_encoded": all_encoded,
            "region_mask_info": region_mask_info,
            "n_vis": n_vis, "n_mask": n_mask,
            "brain_states": all_encoded,
            "subj_logits": None,
        }

    def compute_loss(self, outputs, subject_ids=None):
        pred = outputs["predictions"]
        target = outputs["targets"]
        all_enc = outputs["all_encoded"]

        # Temporal prediction loss
        pred_loss = F.mse_loss(pred, target)

        # Region masking loss
        region_loss = self.region_masker.compute_region_loss(
            outputs.get("region_mask_info"))

        # SIGReg
        B, N, D = all_enc.shape
        x = all_enc.reshape(-1, D)
        var_loss = F.relu(1.0 - x.std(dim=0)).mean()
        x_c = x - x.mean(dim=0, keepdim=True)
        cov = (x_c.T @ x_c) / max(x.shape[0]-1, 1)
        cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D

        query_loss = self.tokenizer.get_query_specialization_loss()

        total = ((1 - self.sigreg_lambda) * pred_loss
                 + self.sigreg_lambda * (var_loss + cov_loss)
                 + self.query_spec_weight * query_loss
                 + self.region_mask_weight * region_loss)

        return {"total": total, "pred": pred_loss, "var": var_loss,
                "cov": cov_loss, "qspec": query_loss, "rmask": region_loss,
                "adv": torch.tensor(0.0, device=pred.device)}

    def update_ema(self): pass
    def set_training_progress(self, p): pass
    def initialize_electrodes(self, e): pass
