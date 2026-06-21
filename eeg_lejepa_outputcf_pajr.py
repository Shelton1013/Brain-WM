"""EEG-LeJEPA + Output-Space Cross-Frequency + Patient-Adversarial Regularization.

Extends `lejepa_outputcf` with a DANN-style domain adversarial head that
predicts patient identity from the pooled encoder output, with a gradient
reversal layer in between. This pushes the encoder to ERASE patient-
identifying features.

Motivation: at TUEG scale (~4000 patients × ~600 trials/patient), the
encoder discovers a "patient-identity shortcut" that trivially solves the
JEPA mask prediction task (same-patient trials look similar → predict
mask = output a constant). Standard anti-collapse regularizers (VICReg,
SIGReg) attack the symptom (low variance) but not the cause (patient-id
encoding). PAJR directly forbids patient-identifiable representations.

Architecture (relative to outputcf):

    EEG → tokenizer → encoder → encoded [B, N, D]
                                    │
                    ┌───────────────┼──────────────┐
                    ▼               ▼              ▼
            (existing JEPA)   (existing CF)   NEW: pool over N
              pred_loss        freq_loss              ▼
                                              grad_reverse(λ_par)
                                                      ▼
                                            patient_disc → logits
                                                      ▼
                                              CE(logits, patient_id)
                                                      ▼
                                                 par_loss

Total loss:
    total = (1-λ)·pred + λ·reg + 1.0·freq + 0.1·qspec + par_weight·par_loss

The adversarial term enters as a positive weight in the total loss; the
gradient reversal layer inverts the sign during backprop, so:
- patient_disc weights: minimize par_loss → become a good classifier
- encoder weights: maximize par_loss → erase patient-identifying signal
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from eeg_lejepa_outputcf import EEGLeJEPAOutputCF


# ============================================================
# Gradient Reversal Layer (Ganin & Lempitsky, ICML 2015)
# ============================================================

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_: float = 1.0):
    return _GradReverse.apply(x, lambda_)


# ============================================================
# Patient Discriminator
# ============================================================

class PatientDiscriminator(nn.Module):
    """MLP that predicts patient identity from a pooled embedding."""

    def __init__(self, d_model: int, n_patients: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_patients),
        )

    def forward(self, x):
        return self.mlp(x)


# ============================================================
# Main model
# ============================================================

class EEGLeJEPAOutputCFPAJR(EEGLeJEPAOutputCF):
    """outputcf + Patient-Adversarial JEPA Regularization."""

    def __init__(
        self,
        n_patients: int = 4000,
        par_lambda: float = 1.0,
        par_weight: float = 0.1,
        par_disc_hidden: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Save PAJR hyperparams as buffers/attrs (par_lambda is used at fwd
        # time; par_weight at compute_loss time)
        self.n_patients = int(n_patients)
        self.par_lambda = float(par_lambda)
        self.par_weight = float(par_weight)
        self.patient_disc = PatientDiscriminator(
            self.d_model, self.n_patients, par_disc_hidden,
        )

    def forward(self, eeg, return_predictions=True):
        # Run the standard outputcf forward
        outputs = super().forward(eeg, return_predictions=return_predictions)

        # PAJR only during training (saves compute at eval/inference)
        if return_predictions and self.training:
            # Pool encoded over tokens → [B, D]
            pooled = outputs["all_encoded"].mean(dim=1)
            # Gradient reversal then discriminator
            pooled_rev = grad_reverse(pooled, self.par_lambda)
            outputs["patient_logits"] = self.patient_disc(pooled_rev)

        return outputs

    def compute_loss(self, outputs, subject_ids=None):
        # Get base losses (pred, reg, freq, qspec, total)
        losses = super().compute_loss(outputs, subject_ids=subject_ids)
        device = outputs["all_encoded"].device

        if "patient_logits" in outputs and subject_ids is not None:
            patient_logits = outputs["patient_logits"]
            # Clamp subject_ids to valid range to be defensive against any
            # mis-assignment in datasets (e.g., new patients beyond n_patients).
            sids = subject_ids.long().clamp_(0, self.n_patients - 1).to(device)
            par_loss = F.cross_entropy(patient_logits, sids)
            losses["par"] = par_loss
            losses["total"] = losses["total"] + self.par_weight * par_loss
            # Also expose discriminator accuracy for monitoring (no grad)
            with torch.no_grad():
                pred = patient_logits.argmax(dim=-1)
                losses["par_acc"] = (pred == sids).float().mean()
        else:
            losses["par"] = torch.tensor(0.0, device=device)
            losses["par_acc"] = torch.tensor(0.0, device=device)

        return losses
