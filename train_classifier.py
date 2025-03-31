from modelUtils import Classifier
from dataUtils import ImageLabelDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

def train_classifier(batch_size, num_workers, num_epochs, learning_rate, model_dir, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = ImageLabelDataset(transform=transform)
    
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model = Classifier().to(device)
    
    # Freeze everything first
    for param in model.model.parameters():
        param.requires_grad = False

    # Unfreeze first conv layer 
    model.model.conv1.requires_grad = True

    # Unfreeze final fully connected layer
    model.model.fc.requires_grad = True


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    accs = []
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()  

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                # For multi-label classification, apply sigmoid and threshold at 0.5
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.numel()  

        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}')
        losses.append(avg_loss)
        accs.append(accuracy)

    torch.save(model.state_dict(), model_dir)
    print(f'Model saved to {model_dir}')
    return losses, accs

if __name__ == "__main__":
    train_classifier(batch_size=32, num_workers=4, num_epochs=10, learning_rate=0.001, model_dir='model.pth')
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
