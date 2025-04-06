from modelUtils import Classifier
from dataUtils import *
# from trainUtils import BCEWithConstraintAndF1Loss
from trainUtils import FocalLoss
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def validate_model(model, val_loader, device, criterion):
    """
    Runs the validation loop once and returns the following metrics:
      - avg_loss: average loss per epoch
      - overall_accuracy: total correct predictions / total predictions
      - per_class_accuracy: accuracy for each class (vector)
      - exact_match: fraction of samples with all labels correctly predicted
      - f1_macro: macro-averaged F1 score
      - f1_micro: micro-averaged F1 score
      - f1_per_class: F1 score for each class (vector)
    """
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_all_preds = []
    val_all_labels = []
    val_exact_match_count = 0
    val_correct_list = []  # to accumulate correct predictions per class

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            # Get binary predictions from probabilities
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            # Overall correct predictions and total number of labels
            val_correct += (predicted == labels).sum().item()
            val_total += labels.numel()
            
            # Per-class correct predictions
            correct_per_class = (predicted == labels).sum(dim=0)
            val_correct_list.append(correct_per_class)
            
            # Accumulate predictions and labels for F1 calculation
            val_all_preds.append(predicted.cpu())
            val_all_labels.append(labels.cpu())
            
            # Count exact match samples (all labels correct)
            val_exact_match_count += (predicted == labels).all(dim=1).sum().item()
    
    # Compute average loss
    avg_loss = val_running_loss / len(val_loader)
    
    # Overall accuracy: total correct / total predictions
    overall_accuracy = val_correct / val_total
    
    # Per-class accuracy: stack and sum over batches, then divide by number of samples
    val_correct_list = torch.stack(val_correct_list).sum(dim=0)
    per_class_accuracy = val_correct_list / len(val_loader.dataset)
    
    # Concatenate predictions and labels (shape: [num_samples, num_classes])
    all_preds_tensor = torch.cat(val_all_preds, dim=0).numpy()
    all_labels_tensor = torch.cat(val_all_labels, dim=0).numpy()
    
    # Exact match: fraction of samples with all labels correct
    exact_match = val_exact_match_count / len(all_preds_tensor)
    
    # F1 scores using scikit-learn (make sure to import f1_score from sklearn.metrics)
    f1_macro = f1_score(all_labels_tensor, all_preds_tensor, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels_tensor, all_preds_tensor, average='micro', zero_division=0)
    f1_per_class = f1_score(all_labels_tensor, all_preds_tensor, average=None, zero_division=0)
    
    return avg_loss, overall_accuracy, per_class_accuracy, exact_match, f1_macro, f1_micro, f1_per_class




def train_classifier(batch_size, num_workers, num_epochs, learning_rate, model_dir, transform,train_split=0.8,patience = 20, finetune = False,model_name="resnet18"):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}',flush=True)

    train_dataset, val_dataset, test_dataset = get_train_val_test_split(transform=transform,train_split=train_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print("*"*10+"Data loaded"+"*"*10,flush=True)
    print(f"Using backbone: {model_name}",flush=True)
    model = Classifier().to(device)
    if finetune:
        print("*" * 10 + "Freezing layers" + "*" * 10, flush=True)
        if model.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            for name, param in model.model.named_parameters():
                if 'conv1' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model.backbone in ['densenet121', 'densenet169', 'densenet201']:
            for name, param in model.model.named_parameters():
                if 'features.conv0' in name or 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    pos_weight_tensor = compute_pos_weight_tensor(device)
    # criterion = BCEWithConstraintAndF1Loss(pos_weight=pos_weight_tensor,penalty_weight=1).to(device)
    criterion = FocalLoss(gamma=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    counter = 0

    train_losses = []
    train_accs_list_list = []#Store the accuracy per class for each epoch
    train_f1s_list_list = []#Store the f1 per class for each epoch
    train_f1_overall_list = []#Store the overall f1 for each epoch
    train_exact_matches = []#Store the exact matches for each epoch
    train_overall_acc_list = []#Store the overall accuracy for each epoch


    val_losses = []
    val_accs_list_list = []
    val_f1s_list_list = []
    val_f1_overall_list = []
    val_exact_matches = []
    best_model_state = None
    val_overall_acc_list = []
    best_f1 = 0

    for epoch in tqdm(range(num_epochs)):
        print(epoch, flush=True)
        # Training Phase
        model.train()
        running_loss = 0.0  # used for calculating average loss per epoch
        train_correct_list = []  # used for calculating accuracy per class
        all_preds_train = []  # used for calculating f1 per class
        all_labels_train = []  # used for calculating f1 per class
        matches = 0  # used for calculating exact matches
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)  # output shape (batch_size, num_classes)
            predicted = (torch.sigmoid(outputs) > 0.3).float()
            # loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # correct per class, used for calculating accuracy per class
            correct_per_class = (predicted == labels).sum(dim=0)
            train_correct_list.append(correct_per_class)
            # list of predicted tensors, each tensor has shape (batch_size, num_classes)
            all_preds_train.append(predicted.detach().cpu())
            all_labels_train.append(labels.detach().cpu())

            # exact matches
            matches += (predicted == labels).all(dim=1).sum().item()

            loss.backward()
            optimizer.step()
        scheduler.step()
    
        # train losses
        train_avg_loss = running_loss / len(train_loader)
        train_losses.append(train_avg_loss)
        # train accuracy per class
        train_correct_list = torch.stack(train_correct_list).sum(dim=0)
        train_acc_list = train_correct_list / len(train_loader.dataset)
        train_accs_list_list.append(train_acc_list)
        print(f"Training accuracy per class: {train_acc_list}", flush=True)
        # overall train accuracy
        all_preds_train = torch.cat(all_preds_train, dim=0)
        all_labels_train = torch.cat(all_labels_train, dim=0)
        overall_accuracy_train = (all_preds_train == all_labels_train).sum().item() / all_labels_train.numel()
        train_overall_acc_list.append(overall_accuracy_train)
        # Train f1 per class
        all_preds_train = all_preds_train.numpy()
        all_labels_train = all_labels_train.numpy()
        f1_per_class_train = f1_score(all_labels_train, all_preds_train, average=None, zero_division=0)  # f1 score for each class
        print(f"Training F1 per class: {f1_per_class_train}", flush=True)
        train_f1s_list_list.append(f1_per_class_train)
        # Compute overall F1 score (micro-average aggregates counts over all classes)
        overall_f1_train = f1_score(all_labels_train, all_preds_train, average='micro', zero_division=0)
        train_f1_overall_list.append(overall_f1_train)
        # train exact matches
        train_exact_matches.append(matches / len(train_loader.dataset))


        # Evaluation Phase
        model.eval()
        running_loss_val = 0.0
        val_correct_list = []
        all_preds_val = []
        all_labels_val = []
        matches_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                # loss
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()
                # correct per class, used for calculating accuracy per class
                predicted = (torch.sigmoid(outputs) > 0.3).float()
                correct_per_class = (predicted == labels).sum(dim=0)
                val_correct_list.append(correct_per_class)
                # list of predicted tensors, each tensor has shape (batch_size, num_classes)
                all_preds_val.append(predicted.cpu())
                all_labels_val.append(labels.cpu())
                # exact matches
                matches_val += (predicted == labels).all(dim=1).sum().item()
        # val losses
        val_avg_loss = running_loss_val / len(val_loader)
        val_losses.append(val_avg_loss)
        # val accuracy per class
        val_correct_list = torch.stack(val_correct_list).sum(dim=0)
        val_acc_list = val_correct_list / len(val_loader.dataset)
        print(f"Val accuracy per class: {val_acc_list}", flush=True)
        val_accs_list_list.append(val_acc_list)
        # overall val accuracy
        all_preds_tensor = torch.cat(all_preds_val, dim=0)
        all_labels_tensor = torch.cat(all_labels_val, dim=0)
        overall_accuracy_val = (all_preds_tensor == all_labels_tensor).sum().item() / all_labels_tensor.numel()
        val_overall_acc_list.append(overall_accuracy_val)
        # val f1 per class
        all_preds_tensor = all_preds_tensor.numpy()
        all_labels_tensor = all_labels_tensor.numpy()
        f1_per_class_val = f1_score(all_labels_tensor, all_preds_tensor, average=None, zero_division=0)
        print(f"Val F1 per class: {f1_per_class_val}", flush=True)
        val_f1s_list_list.append(f1_per_class_val)
        # Compute overall F1 score (micro-average aggregates counts over all classes)
        overall_f1_val = f1_score(all_labels_tensor, all_preds_tensor, average='micro', zero_division=0)
        val_f1_overall_list.append(overall_f1_val)
        # val exact matches
        val_exact_match = matches_val / len(all_preds_tensor)
        val_exact_matches.append(val_exact_match)
        print(f"epoch: {epoch}, train_loss: {train_avg_loss}, val_loss: {val_avg_loss}, "
            f"train_accuracy: {overall_accuracy_train}, val_accuracy: {overall_accuracy_val}, "
            f"train_f1: {overall_f1_train}, val_f1: {overall_f1_val}, "
            f"train_exact_match: {matches / len(train_loader.dataset)}, val_exact_match: {val_exact_match}",
            flush=True)

        if overall_f1_val > best_f1:
            best_f1 = overall_f1_val
            best_model_state = model.state_dict()
            counter = 0  # reset patience
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break

    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'train_losses.txt'), 'w') as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(output_folder, 'train_accs.txt'), 'w') as f:
        for acc in train_accs_list_list:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, 'train_f1s.txt'), 'w') as f:
        for f1 in train_f1s_list_list:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'train_f1_overall.txt'), 'w') as f:
        for f1 in train_f1_overall_list:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'train_exact_matches.txt'), 'w') as f:
        for match in train_exact_matches:
            f.write(f"{match}\n")
    with open(os.path.join(output_folder, 'train_overall_acc.txt'), 'w') as f:
        for acc in train_overall_acc_list:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, 'val_losses.txt'), 'w') as f:
        for loss in val_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(output_folder, 'val_accs.txt'), 'w') as f:
        for acc in val_accs_list_list:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, 'val_f1s.txt'), 'w') as f:
        for f1 in val_f1s_list_list:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'val_f1_overall.txt'), 'w') as f:
        for f1 in val_f1_overall_list:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'val_exact_matches.txt'), 'w') as f:
        for match in val_exact_matches:
            f.write(f"{match}\n")
    with open(os.path.join(output_folder, 'val_overall_acc.txt'), 'w') as f:
        for acc in val_overall_acc_list:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, model_dir), 'wb') as f:
        torch.save(model.state_dict(), f)
    print(f'Model saved to {model_dir}')


    # Test Phase
    avg_loss_test, overall_accuracy_test, per_class_accuracy_test, exact_match_test, f1_macro_test, f1_micro_test, f1_per_class_test = validate_model(model, test_loader, device, criterion)
    print(f"Test Loss: {avg_loss_test}, Test Accuracy: {overall_accuracy_test}, Test Exact Match: {exact_match_test}, Test F1 Macro: {f1_macro_test}, Test F1 Micro: {f1_micro_test}, Test F1 per Class: {f1_per_class_test}, per class accuracy: {per_class_accuracy_test}")
    with open(os.path.join(output_folder, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {avg_loss_test}, Test Accuracy: {overall_accuracy_test}, Test Exact Match: {exact_match_test}, Test F1 Macro: {f1_macro_test}, Test F1 Micro: {f1_micro_test}, Test F1 per Class: {f1_per_class_test}, per class accuracy: {per_class_accuracy_test}")
    with open(os.path.join(output_folder, 'test_per_class_accuracy.txt'), 'w') as f:
        for acc in per_class_accuracy_test:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, 'test_f1_per_class.txt'), 'w') as f:
        for f1 in f1_per_class_test:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'test_f1_macro.txt'), 'w') as f:
        f.write(f"{f1_macro_test}\n")
    with open(os.path.join(output_folder, 'test_f1_micro.txt'), 'w') as f:
        f.write(f"{f1_micro_test}\n")
    with open(os.path.join(output_folder, 'test_exact_match.txt'), 'w') as f:
        f.write(f"{exact_match_test}\n")
    with open(os.path.join(output_folder, 'test_accuracy.txt'), 'w') as f:
        f.write(f"{overall_accuracy_test}\n")
    with open(os.path.join(output_folder, 'test_loss.txt'), 'w') as f:
        f.write(f"{avg_loss_test}\n")
    

    return train_losses, train_overall_acc_list, train_exact_matches, train_f1_overall_list, val_losses, val_overall_acc_list, val_exact_matches, val_f1_overall_list,model,train_dataset, val_dataset, test_dataset




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_losses, train_overall_acc_list, train_exact_matches, train_f1_overall_list, val_losses, val_overall_acc_list, val_exact_matches, val_f1_overall_list, classifier,train,val,test=train_classifier(batch_size=16, num_workers=4, num_epochs=1, transform=transform,learning_rate=0.001, model_dir='model.pth',finetune=True)
    # Define epochs (assuming one metric per epoch)
    epochs = range(1, len(train_losses) + 1)
    results_folder = "results"
    # ------------------------------
    # Plot Losses
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Val Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "losses.png"))
    plt.close()

    # ------------------------------
    # Plot Overall Accuracy
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_overall_acc_list, label="Train Overall Accuracy", marker="o")
    plt.plot(epochs, val_overall_acc_list, label="Validation Overall Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Overall Accuracy")
    plt.title("Train & Val Overall Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "overall_accuracy.png"))
    plt.close()

    # ------------------------------
    # Plot Exact Match Ratio
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_exact_matches, label="Train Exact Matches", marker="o")
    plt.plot(epochs, val_exact_matches, label="Validation Exact Matches", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Exact Match Ratio")
    plt.title("Train & Val Exact Match Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "exact_matches.png"))
    plt.close()

    # ------------------------------
    # Plot Overall F1 Score
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_f1_overall_list, label="Train Overall F1", marker="o")
    plt.plot(epochs, val_f1_overall_list, label="Validation Overall F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Overall F1 Score")
    plt.title("Train & Val Overall F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "overall_f1.png"))
    plt.close()

    print("Plots saved in the 'results' folder.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("finding best thresholds")
    train,val,test = get_train_val_test_split(transform=transform,train_split=0.6)
    train_loader = DataLoader(train, batch_size=16, num_workers=4, shuffle=False)

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

    #test on the test set
    print("Testing on the test set")
    test_loader = DataLoader(test, batch_size=16, num_workers=4, shuffle=False)
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting Predictions"):
            images = images.to(device)
            outputs = classifier(images)
            preds = torch.sigmoid(outputs)
            all_preds_test.append(preds.cpu())
            all_labels_test.append(labels.cpu())
    all_preds_test = torch.cat(all_preds_test, dim=0)   # Shape: (N, num_classes)
    all_labels_test = torch.cat(all_labels_test, dim=0)   # Shape: (N, num_classes)
    # Compute F1 scores using the best thresholds
    all_preds_test = (all_preds_test >= torch.tensor(best_threshold).to(device)).float()
    all_labels_test = all_labels_test.float()
    print(f"Test F1 score: {f1_score(all_labels_test.cpu(), all_preds_test.cpu(), average='macro', zero_division=0)}")
    for i in range(1,200,10):
        train_img,train_labels = train[i]
        test_img,test_labels = test[i]
        train_pred = all_preds_test[i]
        train_pred = train_pred.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        train_img = train_img.cpu().numpy()
        test_img = test_img.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        plt.imshow(train_img[0], cmap='gray')
        plt.title(f"Train Image {i}, Pred: {train_pred}, Labels: {train_labels}")
        plt.close()
        plt.imshow(test_img[0], cmap='gray')
        plt.title(f"Test Image {i}, Pred: {train_pred}, Labels: {test_labels}")
        plt.close()
    



