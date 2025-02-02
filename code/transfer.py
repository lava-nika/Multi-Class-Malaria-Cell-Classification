import torch
import torch.nn as nn
from torchvision import models

class MalariaTransferModel(nn.Module):
    def __init__(self, num_classes=6):
        super(MalariaTransferModel, self).__init__()
        
        # Load the pre-trained ResNet-18 model
        self.model = models.resnet50(pretrained=True)
        
        # Freeze early layers if desired
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)  # New FC layer for 6 classes

    def forward(self, x):
        return self.model(x)
