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

def train_classifier(batch_size, num_workers, num_epochs, learning_rate, model_dir, train_split=0.8):
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
    print("Train dataset length: ", len(train_dataset),flush=True)
    print("Test dataset length: ", len(test_dataset),flush=True)
    print("*"*10+"Loading data"+"*"*10,flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
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
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            i += 1
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            print(loss)
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
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                correct += (predicted == labels).sum().item()
                total += labels.numel()

                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

                # Exact match ratio
                exact_match_count += (predicted == labels).all(dim=1).sum().item()

        # Aggregate and calculate metrics
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

    torch.save(model.state_dict(), model_dir)
    print(f'Model saved to {model_dir}')
    return losses, accs, exact_matches, f1s


if __name__ == "__main__":
    losses,accs=train_classifier(batch_size=16, num_workers=4, num_epochs=10, learning_rate=0.001, model_dir='model.pth')
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
