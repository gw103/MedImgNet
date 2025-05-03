import torch
import torch.nn as nn
import torch.nn.functional as F
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

# def soft_f1_loss(y_pred, y_true, class_weight=None, eps=1e-7):
#     """
#     Differentiable approximation of F1 loss with optional class weighting.
    
#     Args:
#         y_pred: raw logits (before sigmoid), shape (batch_size, num_classes)
#         y_true: ground truth labels (0 or 1), same shape
#         class_weight: Tensor of shape (num_classes,), usually same as pos_weight in BCE
#     """
#     y_pred = torch.sigmoid(y_pred)
#     tp = (y_pred * y_true).sum(dim=0)
#     fp = ((1 - y_true) * y_pred).sum(dim=0)
#     fn = (y_true * (1 - y_pred)).sum(dim=0)

#     soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

#     if class_weight is not None:
#         weighted_f1 = soft_f1 * class_weight
#         return 1 - weighted_f1.sum() / class_weight.sum()
#     else:
#         return 1 - soft_f1.mean()

# class BCEWithConstraintAndF1Loss(nn.Module):
#     def __init__(self, pos_weight=None, penalty_weight=1, f1_weight=10.0):
#         """
#         pos_weight: torch tensor of shape (num_classes,), used for both BCE and F1 loss weighting
#         """
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         self.penalty_weight = penalty_weight
#         self.f1_weight = f1_weight
#         self.pos_weight = pos_weight  

#     def forward(self, logits, targets):
#         base_loss = self.bce(logits, targets)
        
#         f1_loss = soft_f1_loss(
#             logits, 
#             targets, 
#             class_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
#         )

#         probs = torch.sigmoid(logits)  # (batch_size, 15)
#         no_disease_probs = probs[:, 14]
#         other_probs = probs[:, :14]
#         other_sum = other_probs.sum(dim=1)

#         # Constraint 1
#         conflict_1 = no_disease_probs.unsqueeze(1) * other_probs
#         penalty_1 = conflict_1.mean()

#         # Constraint 2
#         conflict_2 = (1 - no_disease_probs) * (1 - torch.clamp(other_sum, max=1.0))
#         penalty_2 = conflict_2.mean()

#         total_penalty = penalty_1 + penalty_2

#         total_loss = base_loss + self.penalty_weight * total_penalty + self.f1_weight * f1_loss
#         return total_loss
counts = {
    "Atelectasis": 15430,
    "Cardiomegaly": 3609,
    "Effusion": 18029,
    "Infiltration": 27765,
    "Mass": 7696,
    "Nodule": 8715,
    "Pneumonia": 1860,
    "Pneumothorax": 7370,
    "Consolidation": 6078,
    "Edema": 2998,
    "Emphysema": 3308,
    "Fibrosis": 2029,
    "Pleural_Thickening": 4630,
    "Hernia": 292,
    "No Finding": 6000
}
total_samples = 111601

# Define the order of classes (your label map)
label_map = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
]

# Compute per-class alpha weights (here, alpha = 1 - (count/total_samples))
alpha_vector = []
for label in label_map:
    freq = counts[label] / total_samples  # frequency in [0, 1]
    alpha_i = 1 - freq
    alpha_vector.append(alpha_i)

# Convert to a torch tensor (shape: [num_classes])
alpha_vector = torch.tensor(alpha_vector, dtype=torch.float)
print("Alpha vector:", alpha_vector)

class FocalLoss(nn.Module):
    def __init__(self, alpha=alpha_vector, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-label classification.
        
        Args:
            alpha (float or list or torch.Tensor): Balancing factor(s) for each class.
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            self.alpha = alpha  # scalar

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits (batch_size, num_classes)
            targets: Binary ground truth (batch_size, num_classes)
        Returns:
            Loss value.
        """
        # Compute binary cross entropy (without reduction)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        # p_t is the probability for the ground truth class for each element
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_term = (1 - p_t) ** self.gamma
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(inputs.device).unsqueeze(0)
        else:
            alpha = self.alpha
        
        loss = alpha * focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


