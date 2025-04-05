import torch
import torch.nn as nn
# import numpy as np

# def soft_f1_loss(y_pred, y_true, class_weights=None, eps=1e-7):
#     y_pred = torch.sigmoid(y_pred)
#     tp = (y_pred * y_true).sum(dim=0)
#     fp = ((1 - y_true) * y_pred).sum(dim=0)
#     fn = (y_true * (1 - y_pred)).sum(dim=0)

#     soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

#     if class_weights is not None:
#         soft_f1 = soft_f1 * class_weights  # weighted F1 per class

#         # Normalize total weight to avoid scaling up the loss
#         return 1 - (soft_f1.sum() / class_weights.sum())
#     else:
#         return 1 - soft_f1.mean()
# class BCEWithConstraintAndF1Loss(nn.Module):
#     def __init__(self, pos_weight=None, penalty_weight=1, f1_weight=10.0):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         self.penalty_weight = penalty_weight
#         self.f1_weight = f1_weight

#         # Hardcoded F1 scores
#         f1_scores = np.array([
#             0.2682318, 0.09024319, 0.37409157, 0.36400793, 0.16611018,
#             0.12420671, 0.00913242, 0.20619571, 0., 0.17163967,
#             0.11333062, 0.06787998, 0.11300476, 0., 0.67911867
#         ])
#         f1_weights = 1.0 / (f1_scores + 1e-4)
#         f1_weights = f1_weights / f1_weights.sum()
#         self.register_buffer('f1_class_weights', torch.tensor(f1_weights, dtype=torch.float32))

#     def forward(self, logits, targets):
#         base_loss = self.bce(logits, targets)
#         f1_loss = soft_f1_loss(logits, targets, class_weights=self.f1_class_weights)

#         probs = torch.sigmoid(logits)
#         no_disease_probs = probs[:, 14]
#         other_probs = probs[:, :14]
#         other_sum = other_probs.sum(dim=1)

#         conflict_1 = no_disease_probs.unsqueeze(1) * other_probs
#         penalty_1 = conflict_1.mean()

#         conflict_2 = (1 - no_disease_probs) * (1 - torch.clamp(other_sum, max=1.0))
#         penalty_2 = conflict_2.mean()

#         total_penalty = penalty_1 + penalty_2
#         total_loss = base_loss + self.penalty_weight * total_penalty + self.f1_weight * f1_loss
#         return total_loss

def soft_f1_loss(y_pred, y_true, class_weight=None, eps=1e-7):
    """
    Differentiable approximation of F1 loss with optional class weighting.
    
    Args:
        y_pred: raw logits (before sigmoid), shape (batch_size, num_classes)
        y_true: ground truth labels (0 or 1), same shape
        class_weight: Tensor of shape (num_classes,), usually same as pos_weight in BCE
    """
    y_pred = torch.sigmoid(y_pred)
    tp = (y_pred * y_true).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    if class_weight is not None:
        weighted_f1 = soft_f1 * class_weight
        return 1 - weighted_f1.sum() / class_weight.sum()
    else:
        return 1 - soft_f1.mean()

class BCEWithConstraintAndF1Loss(nn.Module):
    def __init__(self, pos_weight=None, penalty_weight=1, f1_weight=10.0):
        """
        pos_weight: torch tensor of shape (num_classes,), used for both BCE and F1 loss weighting
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.penalty_weight = penalty_weight
        self.f1_weight = f1_weight
        self.pos_weight = pos_weight  

    def forward(self, logits, targets):
        base_loss = self.bce(logits, targets)
        
        f1_loss = soft_f1_loss(
            logits, 
            targets, 
            class_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )

        probs = torch.sigmoid(logits)  # (batch_size, 15)
        no_disease_probs = probs[:, 14]
        other_probs = probs[:, :14]
        other_sum = other_probs.sum(dim=1)

        # Constraint 1
        conflict_1 = no_disease_probs.unsqueeze(1) * other_probs
        penalty_1 = conflict_1.mean()

        # Constraint 2
        conflict_2 = (1 - no_disease_probs) * (1 - torch.clamp(other_sum, max=1.0))
        penalty_2 = conflict_2.mean()

        total_penalty = penalty_1 + penalty_2

        total_loss = base_loss + self.penalty_weight * total_penalty + self.f1_weight * f1_loss
        return total_loss
