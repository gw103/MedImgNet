import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm as tqdm

from modelUtils import Classifier
from dataUtils import get_train_val_test_split

classifier = Classifier()
classifier.load_state_dict(torch.load("results/model.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
classifier.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_split = 0.6
train_dataset, _, _ = get_train_val_test_split(transform=transform, train_split=train_split)
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=False)


thresholds = np.linspace(0.0, 1.0, 20)
num_classes = 15
best_threshold = np.zeros(num_classes)
best_f1 = np.zeros(num_classes)

for i in tqdm(range(num_classes)):
    for thres in tqdm(thresholds):
        total_tp, total_fp, total_fn = 0, 0, 0
        for batch in train_loader:
            images, labels = batch
            images = images.to(next(classifier.parameters()).device)
            labels = labels.to(next(classifier.parameters()).device)
            with torch.no_grad():
                outputs = classifier(images)
                probs = torch.sigmoid(outputs)
            preds = (probs[:, i] > thres).float()
            labeli = labels[:, i].float()
            
            tp = torch.sum(preds * labeli).item()
            fp = torch.sum(preds * (1 - labeli)).item()
            fn = torch.sum((1 - preds) * labeli).item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

        denominator = (2 * total_tp + total_fp + total_fn)
        if denominator == 0:
            current_f1 = 0.0
        else:
            current_f1 = 2 * total_tp / denominator

        if current_f1 > best_f1[i]:
            best_f1[i] = current_f1
            best_threshold[i] = thres

for i in range(num_classes):
    print(f"Class {i}: Best threshold = {best_threshold[i]:.3f}, F1 score = {best_f1[i]:.3f}")
