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
def validate_model(model, val_loader, device, criterion):
    """
    Runs the validation loop once and returns scalar averages for the metrics.
    """
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_all_preds = []
    val_all_labels = []
    val_exact_match_count = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            val_correct += (predicted == labels).sum().item()
            val_total += labels.numel()

            val_all_preds.append(predicted.cpu())
            val_all_labels.append(labels.cpu())
            # Count samples where all labels match exactly
            val_exact_match_count += (predicted == labels).all(dim=1).sum().item()

    avg_loss = val_running_loss / len(val_loader)
    accuracy = val_correct / val_total
    all_preds_tensor = torch.cat(val_all_preds, dim=0).numpy()
    all_labels_tensor = torch.cat(val_all_labels, dim=0).numpy()
    exact_match = val_exact_match_count / len(all_preds_tensor)
    f1 = f1_score(all_labels_tensor, all_preds_tensor, average='macro', zero_division=0)

    return avg_loss, accuracy, exact_match, f1




def train_classifier(batch_size, num_workers, num_epochs, learning_rate, model_dir, train_split=0.6):
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
 
    losses = []
    accs = []
    exact_matches = []
    f1s = []

    for epoch in tqdm(range(num_epochs)):
        print(epoch,flush=True)
        # Training Phase
        model.train()
        running_loss = 0.0
        i = 0
        for images, labels in tqdm(train_loader, desc="Training"):
            i += 1
            images = images.to(device)
            
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if i % 100 == 0:
                print(f'Batch {i}: Loss {loss.item():.4f}', flush=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluation Phase
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        exact_match_count = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="testing", leave=False):
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                correct += (predicted == labels).sum().item()
                total += labels.numel()

                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
            
                exact_match_count += (predicted == labels).all(dim=1).sum().item()
        accuracy = correct / total
        all_preds_tensor = torch.cat(all_preds, dim=0).numpy()
        all_labels_tensor = torch.cat(all_labels, dim=0).numpy()

        exact_match = exact_match_count / len(all_preds_tensor)
        f1 = f1_score(all_labels_tensor, all_preds_tensor, average='macro', zero_division=0)

        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Exact Match: {exact_match:.4f} | F1: {f1:.4f}',flush=True)
        
        losses.append(avg_loss)
        accs.append(accuracy)
        exact_matches.append(exact_match)
        f1s.append(f1)
    with open('losses.txt', 'w') as f:
        for item in losses:
            f.write("%s\n" % item)
    with open('accs.txt', 'w') as f:
        for item in accs:
            f.write("%s\n" % item)
    with open('exact_matches.txt', 'w') as f:
        for item in exact_matches:
            f.write("%s\n" % item)
    with open('f1s.txt', 'w') as f:
        for item in f1s:
            f.write("%s\n" % item)
    

    torch.save(model.state_dict(), model_dir)
    print(f'Model saved to {model_dir}')
    val_loss, val_accuracy, val_exact_match, val_f1 = validate_model(model, val_loader, device, criterion)
    print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | '
          f'Validation Exact Match: {val_exact_match:.4f} | Validation F1: {val_f1:.4f}', flush=True)
    
    with open('val_loss.txt', 'w') as f:
        f.write(f"{val_loss}\n")
    with open('val_acc.txt', 'w') as f:
        f.write(f"{val_accuracy}\n")
    with open('val_exact_match.txt', 'w') as f:
        f.write(f"{val_exact_match}\n")
    with open('val_f1.txt', 'w') as f:
        f.write(f"{val_f1}\n")
    return losses, accs, exact_matches, f1s


if __name__ == "__main__":
    losses, accs, exact_matches, f1s=train_classifier(batch_size=16, num_workers=4, num_epochs=10, learning_rate=0.001, model_dir='model.pth')
    x = [i for i in range(10)]
    plt.plot(x, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')
    plt.show()
    plt.plot(x, accs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig('accuracy.png')
    plt.show()
    plt.plot(x, exact_matches)
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match')
    plt.title('Exact Match')
    plt.savefig('exact_match.png')
    plt.show()
    plt.plot(x, f1s)
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('F1')
    plt.savefig('f1.png')
    plt.show()
