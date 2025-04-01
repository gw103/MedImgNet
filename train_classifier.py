from modelUtils import Classifier
from dataUtils import ImageLabelDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import os
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




def train_classifier(batch_size, num_workers, num_epochs, learning_rate, model_dir, train_split=0.6,patience = 5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}',flush=True)

    dataset = ImageLabelDataset(transform=transform)
    print("Dataset length: ", len(dataset),flush=True)
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_len = len(test_dataset)
    val_len = test_len // 2
    test_len = test_len - val_len  

    val_dataset, test_dataset = random_split(test_dataset, [val_len, test_len])

    print("Train dataset length: ", len(train_dataset),flush=True)
    print("Test dataset length: ", len(test_dataset),flush=True)
    print("Val dataset length: ", len(val_dataset),flush=True)
    print("*"*10+"Loading data"+"*"*10,flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print("*"*10+"Data loaded"+"*"*10,flush=True)
    model = Classifier().to(device)
    print("*"*10+"Freezing layers"+"*"*10,flush=True)
    for name, param in model.model.named_parameters():
        if 'conv1' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
 

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    train_losses = []
    train_accs_list_list = []#Store the accuracy per class for each epoch
    train_f1s_list_list = []#Store the f1 per class for each epoch
    train_f1_overall_list = []#Store the overall f1 for each epoch
    train_exact_matches = []#Store the exact matches for each epoch
    train_overall_acc_list = []#Store the overall accuracy for each epoch


    test_losses = []
    test_accs_list_list = []
    test_f1s_list_list = []
    test_f1_overall_list = []
    test_exact_matches = []
    best_model_state = None
    test_overall_acc_list = []
    best_f1 = 0


    for epoch in tqdm(range(num_epochs)):
        print(epoch,flush=True)
        # Training Phase
        model.train()
        running_loss = 0.0 #used for calculating average loss per epoch
        train_correct_list = [] #used for calculating accuracy per class
        all_preds_train = [] #used for calculating f1 per class
        all_labels_train = [] #used for calculating f1 per class
        matches = 0 #used for calculating exact matches
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images) # output shape (batch_size, num_classes)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            #loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            #correct per class, used for calculating accuracy per class
            correct_per_class = (predicted == labels).sum(dim=0) 
            train_correct_list.append(correct_per_class)
            # list of predicted tensors, each tensor has shape (batch_size, num_classes)
            all_preds_train.append(predicted.detach().cpu())
            all_labels_train.append(labels.detach().cpu())
            # Compute overall training accuracy (individual label level)
            overall_accuracy_train = (all_preds_tensor == all_labels_tensor).sum().item() / all_labels_tensor.numel()
            train_overall_acc_list.append(overall_accuracy_train)
            print(f"Overall Training Accuracy: {overall_accuracy_train}", flush=True)
            #exact matches
            matches += (predicted == labels).all(dim=1).sum().item()

            loss.backward()
            optimizer.step()
            
        #train losses
        train_avg_loss = running_loss / len(train_loader)
        train_losses.append(train_avg_loss)
        #train accuracy per class
        train_correct_list = torch.stack(train_correct_list).sum(dim=0)
        train_acc_list = train_correct_list / len(train_loader.dataset)
        train_accs_list_list.append(train_acc_list)
        print(f"Training accuracy per class: {train_acc_list}",flush=True)
        #Train f1 per class
        all_preds_train = torch.cat(all_preds_train, dim=0).numpy() # shape (num_samples, num_classes)
        all_labels_train = torch.cat(all_labels_train, dim=0).numpy()
        f1_per_class_train = f1_score(all_labels_train, all_preds_train, average=None, zero_division=0) # f1 score for each classes, shape (num_classes,)
        print(f"Training F1 per class: {f1_per_class_train}", flush=True)
        train_f1s_list_list.append(f1_per_class_train)
        # Compute overall F1 score (micro-average aggregates counts over all classes)
        overall_f1_train = f1_score(all_labels_train, all_preds_train, average='micro', zero_division=0)
        train_f1_overall_list.append(overall_f1_train)
        #train exact matches
        train_exact_matches.append(matches / len(train_loader.dataset))


        # Evaluation Phase
        # running_loss = 0.0 #used for calculating average loss per epoch
        # train_correct_list = [] #used for calculating accuracy per class
        # all_preds_train = [] #used for calculating f1 per class
        # all_labels_train = [] #used for calculating f1 per class
        # matches = 0 #used for calculating exact matches
        model.eval()
        running_loss_test = 0.0
        test_correct_list = []
        all_preds_test = []
        all_labels_test = []
        matches_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                #loss
                loss = criterion(outputs, labels)
                running_loss_test += loss.item()
                #correct per class, used for calculating accuracy per class
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_per_class = (predicted == labels).sum(dim=0)
                test_correct_list.append(correct_per_class)
                # list of predicted tensors, each tensor has shape (batch_size, num_classes)
                all_preds_test.append(predicted.cpu())
                all_labels_test.append(labels.cpu())
                #exact matches
                matches_test += (predicted == labels).all(dim=1).sum().item()
        #test losses
        test_avg_loss = running_loss_test / len(test_loader)
        test_losses.append(test_avg_loss)
        #test accuracy per class
        test_correct_list = torch.stack(test_correct_list).sum(dim=0)
        test_acc_list = test_correct_list / len(test_loader.dataset)
        print(f"Test accuracy per class: {test_acc_list}",flush=True)
        test_accs_list_list.append(test_acc_list)
        #overall test accuracy
        all_preds_tensor = torch.cat(all_preds_test, dim=0)
        all_labels_tensor = torch.cat(all_labels_test, dim=0)
        overall_accuracy_test = (all_preds_tensor == all_labels_tensor).sum().item() / all_labels_tensor.numel()
        test_overall_acc_list.append(overall_accuracy_test)
        #test f1 per class
        all_preds_tensor = all_preds_tensor.numpy()
        all_labels_tensor = all_labels_tensor.numpy()
        f1_per_class_test = f1_score(all_labels_tensor, all_preds_tensor, average='macro', zero_division=0)
        print(f"Test F1 per class: {f1_per_class_test}", flush=True)
        test_f1s_list_list.append(f1_per_class_test)
        # Compute overall F1 score (micro-average aggregates counts over all classes)
        overall_f1_test = f1_score(all_labels_tensor, all_preds_tensor, average='micro', zero_division=0)
        test_f1_overall_list.append(overall_f1_test)
        #test exact matches
        test_exact_match = matches_test/ len(all_preds_tensor)
        test_exact_matches.append(test_exact_match)
        print(f"epoch: {epoch}, train_loss: {train_avg_loss}, test_loss: {test_avg_loss}, train_accuracy: {overall_accuracy_train}, test_accuracy: {overall_accuracy_test}, train_f1: {overall_f1_train}, test_f1: {overall_f1_test}, train_exact_match: {matches / len(train_loader.dataset)}, test_exact_match: {test_exact_match}",flush=True)

        if overall_f1_test > best_f1:
            best_f1 = overall_f1_test
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
    with open(os.path.join(output_folder, 'test_losses.txt'), 'w') as f:
        for loss in test_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(output_folder, 'test_accs.txt'), 'w') as f:
        for acc in test_accs_list_list:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, 'test_f1s.txt'), 'w') as f:
        for f1 in test_f1s_list_list:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'test_f1_overall.txt'), 'w') as f:
        for f1 in test_f1_overall_list:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'test_exact_matches.txt'), 'w') as f:
        for match in test_exact_matches:
            f.write(f"{match}\n")
    with open(os.path.join(output_folder, 'test_overall_acc.txt'), 'w') as f:
        for acc in test_overall_acc_list:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, model_dir), 'wb') as f:
        torch.save(model.state_dict(), f)
    print(f'Model saved to {model_dir}')

    # Validation Phase
    avg_loss_val, overall_accuracy_val, per_class_accuracy_val, exact_match_val, f1_macro_val, f1_micro_val, f1_per_class_val = validate_model(model, val_loader, device, criterion)
    print(f"Validation Loss: {avg_loss_val}, Validation Accuracy: {overall_accuracy_val}, Validation Exact Match: {exact_match_val}, Validation F1 Macro: {f1_macro_val}, Validation F1 Micro: {f1_micro_val}, Validation F1 per Class: {f1_per_class_val},per class accuracy: {per_class_accuracy_val}")
    with open(os.path.join(output_folder, 'validation_results.txt'), 'w') as f:
        f.write(f"Validation Loss: {avg_loss_val}, Validation Accuracy: {overall_accuracy_val}, Validation Exact Match: {exact_match_val}, Validation F1 Macro: {f1_macro_val}, Validation F1 Micro: {f1_micro_val}, Validation F1 per Class: {f1_per_class_val},per class accuracy: {per_class_accuracy_val}")
    with open(os.path.join(output_folder, 'validation_per_class_accuracy.txt'), 'w') as f:
        for acc in per_class_accuracy_val:
            f.write(f"{acc}\n")
    with open(os.path.join(output_folder, 'validation_f1_per_class.txt'), 'w') as f:
        for f1 in f1_per_class_val:
            f.write(f"{f1}\n")
    with open(os.path.join(output_folder, 'validation_f1_macro.txt'), 'w') as f:
        f.write(f"{f1_macro_val}\n")
    with open(os.path.join(output_folder, 'validation_f1_micro.txt'), 'w') as f:
        f.write(f"{f1_micro_val}\n")
    with open(os.path.join(output_folder, 'validation_exact_match.txt'), 'w') as f:
        f.write(f"{exact_match_val}\n")
    with open(os.path.join(output_folder, 'validation_accuracy.txt'), 'w') as f:
        f.write(f"{overall_accuracy_val}\n")
    with open(os.path.join(output_folder, 'validation_loss.txt'), 'w') as f:
        f.write(f"{avg_loss_val}\n")
    return train_losses, train_overall_acc_list, train_exact_matches, train_f1_overall_list, test_losses, test_overall_acc_list, test_exact_matches, test_f1_overall_list




if __name__ == "__main__":
    train_losses, train_overall_acc_list, train_exact_matches, train_f1_overall_list, test_losses, test_overall_acc_list, test_exact_matches, test_f1_overall_list=train_classifier(batch_size=16, num_workers=4, num_epochs=10, learning_rate=0.001, model_dir='model.pth')
    # Define epochs (assuming one metric per epoch)
    epochs = range(1, len(train_losses) + 1)
    results_folder = "results"
    # ------------------------------
    # Plot Losses
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "losses.png"))
    plt.close()

    # ------------------------------
    # Plot Overall Accuracy
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_overall_acc_list, label="Train Overall Accuracy", marker="o")
    plt.plot(epochs, test_overall_acc_list, label="Test Overall Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Overall Accuracy")
    plt.title("Train & Test Overall Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "overall_accuracy.png"))
    plt.close()

    # ------------------------------
    # Plot Exact Match Ratio
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_exact_matches, label="Train Exact Matches", marker="o")
    plt.plot(epochs, test_exact_matches, label="Test Exact Matches", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Exact Match Ratio")
    plt.title("Train & Test Exact Match Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "exact_matches.png"))
    plt.close()

    # ------------------------------
    # Plot Overall F1 Score
    # ------------------------------
    plt.figure()
    plt.plot(epochs, train_f1_overall_list, label="Train Overall F1", marker="o")
    plt.plot(epochs, test_f1_overall_list, label="Test Overall F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Overall F1 Score")
    plt.title("Train & Test Overall F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "overall_f1.png"))
    plt.close()

    print("Plots saved in the 'results' folder.")
