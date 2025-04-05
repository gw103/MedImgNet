import torch
import torch.nn as nn

import torch
import torch.nn as nn



import torch
import torch.nn as nn

def soft_f1_loss(y_pred, y_true, eps=1e-7):
    """
    Differentiable approximation of F1 loss.
    y_pred: raw logits (before sigmoid), shape (batch_size, num_classes)
    y_true: ground truth labels (0 or 1), same shape
    """
    y_pred = torch.sigmoid(y_pred)
    tp = (y_pred * y_true).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return 1 - soft_f1.mean()  # minimize 1 - F1

class BCEWithConstraintAndF1Loss(nn.Module):
    def __init__(self, pos_weight=None, penalty_weight=1, f1_weight=1.0):
        """
        penalty_weight: how strongly to enforce class 14 exclusivity logic
        f1_weight: how much to prioritize soft F1 optimization
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.penalty_weight = penalty_weight
        self.f1_weight = f1_weight

    def forward(self, logits, targets):
        base_loss = self.bce(logits, targets)
        f1_loss = soft_f1_loss(logits, targets)

        probs = torch.sigmoid(logits)  # (batch_size, 15)

        no_disease_probs = probs[:, 14]      # shape: (batch_size,)
        other_probs = probs[:, :14]          # shape: (batch_size, 14)
        other_sum = other_probs.sum(dim=1)   # shape: (batch_size,)

        ### Constraint 1: if no_disease_prob is high, other_probs should be low
        conflict_1 = no_disease_probs.unsqueeze(1) * other_probs  # shape: (batch_size, 14)
        penalty_1 = conflict_1.mean()

        ### Constraint 2: if no_disease_prob is low, other_probs should have at least one active
        conflict_2 = (1 - no_disease_probs) * (1 - torch.clamp(other_sum, max=1.0))  # shape: (batch_size,)
        penalty_2 = conflict_2.mean()

        total_penalty = penalty_1 + penalty_2

        total_loss = base_loss + self.penalty_weight * total_penalty + self.f1_weight * f1_loss
        return total_loss
