import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transfer import MalariaTransferModel  # Import the modified model
from PIL import Image
import time

# Hyperparameters
num_classes = 6
num_epochs = 5
batch_size = 32
learning_rate = 0.0001

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size required by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data directories
train_data_path = '../data/train'  
test_data_path = '../data/test/unlabeled' 

# Load training data
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define custom dataset class for test data
class UnlabeledDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(images_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# Load the test dataset without labels
test_dataset = UnlabeledDataset(images_folder=test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, criterion, and optimizer
model = MalariaTransferModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Track training time
total_training_time = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # End timing for the epoch
    epoch_time = time.time() - start_time
    total_training_time += epoch_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f} seconds")

final_training_error = avg_loss
print(f"Final Training Error: {final_training_error:.4f}")
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Generate predictions for the test set
model.eval()
predictions = []
filenames = []

with torch.no_grad():
    for images, image_files in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        filenames.extend(image_files)  

# Convert class indices to names
class_names = train_dataset.classes
predicted_labels = [class_names[pred] for pred in predictions]

# Save predictions to CSV
df = pd.DataFrame({"Input": filenames, "Class": predicted_labels})
df.to_csv("transfer_predictions.csv", index=False)
print("Predictions saved to transfer_predictions.csv")

