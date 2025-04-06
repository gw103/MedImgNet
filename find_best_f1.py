import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from modelUtils import Classifier
from dataUtils import get_train_val_test_split
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# Set device and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = Classifier().to(device)
classifier.load_state_dict(torch.load("results/model.pth", map_location=device))
classifier.eval()

# Define transforms and dataset loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_split = 0.6
train_dataset, _, _ = get_train_val_test_split(transform=transform, train_split=train_split)
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=False)

# ===== Step 1: Collect predictions and labels for the entire dataset =====
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in tqdm(train_loader, desc="Collecting Predictions"):
        images = images.to(device)
        outputs = classifier(images)
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds, dim=0)   # Shape: (N, num_classes)
all_labels = torch.cat(all_labels, dim=0)   # Shape: (N, num_classes)

# ===== Step 2: Vectorized threshold search =====
thresholds = np.linspace(0.0, 1.0, 20)      # Candidate thresholds
num_classes = 15
best_threshold = np.zeros(num_classes)
best_f1 = np.zeros(num_classes)

# Convert thresholds to torch tensor for vectorized operations
thresholds_tensor = torch.tensor(thresholds, dtype=torch.float)

# For each class, compute F1 scores for all thresholds at once
for i in tqdm(range(num_classes)):
    # Get predictions and ground truth for class i
    p = all_preds[:, i]       # (N,)
    gt = all_labels[:, i]     # (N,)
    
    # Expand dims: thresholds_tensor shape (T, 1) vs. p shape (N,)
    # Compare each threshold with all predictions
    preds = (p.unsqueeze(0) >= thresholds_tensor.unsqueeze(1)).float()  # Shape: (T, N)
    
    # Compute true positives, false positives, and false negatives for each threshold
    tp = (preds * gt.unsqueeze(0)).sum(dim=1)
    fp = (preds * (1 - gt.unsqueeze(0))).sum(dim=1)
    fn = ((1 - preds) * gt.unsqueeze(0)).sum(dim=1)
    
    # Compute F1 score vector (add a tiny constant to denominator to avoid division by zero)
    denominator = 2 * tp + fp + fn + 1e-7
    f1_scores = 2 * tp / denominator
    
    # Choose the threshold with the best F1 score
    best_idx = torch.argmax(f1_scores)
    best_f1[i] = f1_scores[best_idx].item()
    best_threshold[i] = thresholds_tensor[best_idx].item()


for i in range(num_classes):
    print(f"Class {i}: Best threshold = {best_threshold[i]:.3f}, F1 score = {best_f1[i]:.3f}")
