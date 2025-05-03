import torch
import torch.nn as nn
import torchvision
class Classifier(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=9, pretrained=False):
        super(Classifier, self).__init__()
        self.backbone = backbone.lower()
        if self.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            resnet_func = getattr(torchvision.models, self.backbone)
            weights = None
            if pretrained:
                if self.backbone == 'resnet50':
                    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet_func(weights=weights)
            # Modify first conv layer to accept 1-channel input.
            self.model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False
            )
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        elif self.backbone in ['densenet121', 'densenet169', 'densenet201']:
            densenet_func = getattr(torchvision.models, self.backbone)
            weights = None
            if pretrained:
                if self.backbone == 'densenet121':
                    weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
                elif self.backbone == 'densenet169':
                    weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
                elif self.backbone == 'densenet201':
                    weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
            self.model = densenet_func(weights=weights)
            # Modify first conv layer.
            self.model.features.conv0 = nn.Conv2d(
                in_channels=1,
                out_channels=self.model.features.conv0.out_channels,
                kernel_size=self.model.features.conv0.kernel_size,
                stride=self.model.features.conv0.stride,
                padding=self.model.features.conv0.padding,
                bias=False
            )
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError("Unsupported backbone.")
    
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