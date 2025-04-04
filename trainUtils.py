import torch
import torch.nn as nn

import torch
import torch.nn as nn

class BCEWithConstraintLoss(nn.Module):
    def __init__(self, pos_weight=None, penalty_weight=10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        base_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)  # (batch_size, 15)

        no_disease_probs = probs[:, 14]      # shape: (batch_size,)
        other_probs = probs[:, :14]          # shape: (batch_size, 14)
        other_sum = other_probs.sum(dim=1)   # shape: (batch_size,)

        ### Constraint 1: if no_disease_prob is high, other_probs should be low
        conflict_1 = no_disease_probs.unsqueeze(1) * other_probs  # shape: (batch_size, 14)
        penalty_1 = conflict_1.mean()

        ### Constraint 2: if no_disease_prob is low, other_probs should have at least one active
        # Use sigmoid values to softly check "at least one > 0"
        # Here we penalize the case where both: no_disease_prob is low AND sum of other probs is low
        conflict_2 = (1 - no_disease_probs) * (1 - torch.clamp(other_sum, max=1.0))  # shape: (batch_size,)
        penalty_2 = conflict_2.mean()

        total_penalty = penalty_1 + penalty_2
        total_loss = base_loss + self.penalty_weight * total_penalty
        return total_loss

