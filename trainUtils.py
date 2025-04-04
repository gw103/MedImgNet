import torch
import torch.nn as nn

class BCEWithConstraintLoss(nn.Module):
    def __init__(self, pos_weight=None, penalty_weight=10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        base_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        no_disease_probs = probs[:, 14]
        other_probs = probs[:, :14] 

        conflict = no_disease_probs.unsqueeze(1) * other_probs  # shape: (batch_size, 14)
        penalty = conflict.mean()

        total_loss = base_loss + self.penalty_weight * penalty
        return total_loss
