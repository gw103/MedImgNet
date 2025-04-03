import torch
import torch.nn as nn
import torchvision


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = torchvision.models.resnet101(weights=None)
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        # Change the final layer to output 14 classes

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 15)
        )

    def forward(self, x):
        return self.model(x)
    
class Regressor(nn.Module):
    def __init__(self, classifer_dir=None, classifier=None):
        super(Regressor, self).__init__()
        self.classifier = load_state_dict(torch.load(classifer_dir))
        self.classifier.eval() 
        #Can adjust here in the future 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.regressor_fc = nn.Sequential(
            nn.Linear(14 + 10, 128), # Input: 14-dimensional vector + 10-dimensional vector
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 14)  # Output: 14-dimensional vector
        )
        

    def forward(self, x, classifier_output):
        # Concatenate classifier output with image features
        raw = x
        x = self.conv_layers(raw)
        res = self.classifier(raw)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, res], dim=1)
        x = self.regressor_fc(x)
        return x