import logging
import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import timm

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Custom Dataset for Multi-Label Classification
class VehicleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.df.columns[1:]  # Skip 'filename' column
        # Convert label columns to numeric (float) to handle potential string '0'/'1'
        self.df[self.labels] = self.df[self.labels].apply(pd.to_numeric, errors='coerce')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]  # 'filename'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        labels = torch.tensor(self.df.iloc[idx, 1:].to_numpy(dtype='float32'), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
base_dir = '/home/masaya/datasets'
train_dataset = VehicleDataset(
    csv_file=os.path.join(base_dir, 'train', '_classes.csv'),
    img_dir=os.path.join(base_dir, 'train'),
    transform=transform,
)
valid_dataset = VehicleDataset(
    csv_file=os.path.join(base_dir, 'valid', '_classes.csv'),
    img_dir=os.path.join(base_dir, 'valid'),
    transform=transform,
)
test_dataset = VehicleDataset(
    csv_file=os.path.join(base_dir, 'test', '_classes.csv'),
    img_dir=os.path.join(base_dir, 'test'),
    transform=transform,
)

BATCH_SIZE = 128

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

logging.info(
    'Loaded %d training, %d validation, and %d test samples.',
    len(train_dataset),
    len(valid_dataset),
    len(test_dataset),
)
logging.info('Using batch size: %d', BATCH_SIZE)

# Define model
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.labels)  # Dynamically set based on dataset
model = MultiLabelModel(num_classes=num_classes).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

logging.info('Using device: %s', device)
logging.info('Model has %d output classes.', num_classes)

# Training loop
num_epochs = 70
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    logging.info('Starting epoch %d/%d', epoch + 1, num_epochs)
    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        if batch_idx % 10 == 0 or batch_idx == len(train_loader):
            logging.info('Epoch %d/%d | Batch %d/%d | Batch Loss: %.4f',
                         epoch + 1,
                         num_epochs,
                         batch_idx,
                         len(train_loader),
                         loss.item())
    
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
    
    valid_loss /= len(valid_loader.dataset)
    logging.info('Completed epoch %d/%d | Train Loss: %.4f | Valid Loss: %.4f',
                 epoch + 1,
                 num_epochs,
                 train_loss,
                 valid_loss)

# Final test evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

test_loss /= len(test_loader.dataset)
logging.info('Test Loss: %.4f', test_loss)

# Save the model
model_path = 'multilabel_vehicle_model_production.pth'
torch.save(model.state_dict(), model_path)
logging.info('Saved trained model to %s', model_path)

# Inference example (on a single image)
def predict_image(img_path, model, transform, device):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs.cpu().numpy()[0]

# Example usage
# img_path = os.path.join(base_dir, 'test', 'some_image.jpg')  # Replace with an actual test image path
# predictions = predict_image(img_path, model, transform, device)
# labels = train_dataset.labels
# for label, prob in zip(labels, predictions):
#     if prob > 0.5:  # Threshold for positive labels
#         print(f'{label}: {prob:.4f}')
