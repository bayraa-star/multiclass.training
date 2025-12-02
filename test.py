import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import timm

# Define the model class (must match the training code)
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load label names from the training CSV
base_dir = '/opt/roboflow/vehicle-brand-color-side-3'
csv_file = os.path.join(base_dir, 'train', '_classes.csv')
df = pd.read_csv(csv_file)
labels = df.columns[1:]  # Skip 'filename' column
num_classes = len(labels)

# Initialize model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiLabelModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('multilabel_vehicle_model_epch_70K.pth'))  # Load your saved model

# Function to determine category based on label
def get_category(label):
    # Common colors
    colors = {'black', 'blue', 'red', 'brown', 'gray', 'grey', 'green', 'grey', 'red', 'white', 'yellow'}
    # Common vehicle types
    vehicle_types = {'car', 'truck'}
    # Sides
    sides = {'rear_side', 'front_side'}

    if label.lower() in colors:
        return 'color'
    elif label.lower() in vehicle_types:
        return 'vehicle_type'
    elif label.lower() in sides or label.lower().endswith('_side'):
        return 'side'
    else:
        return 'mark'  # Assuming remaining are brands/models like toyota_prius

# Inference function (same as before)
def predict_image(img_path, model, transform, device):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs.cpu().numpy()[0]

# Run prediction on your image.png (adjust the path if needed)
img_path = 'test.png'  # Replace with the full path if needed, e.g., '/path/to/test.png'
predictions = predict_image(img_path, model, transform, device)

# Print only predicted labels with probabilities > 0.5 in the desired format
print("Predicted labels for image.png:")
for label, prob in zip(labels, predictions):
    if prob > 0.5:
        category = get_category(label)
        print(f'{category}: {label}: {prob:.4f}')