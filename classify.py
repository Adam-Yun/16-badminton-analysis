import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from collections import deque

"""
CLASSIFY MODULE: Rally / Non-Rally Scene Classifier for Badminton Analysis

1. DATASET STRUCTURE:
Organize your training data as follows:
dataset/
    train/
        rally/
            frame_0001.jpg
            ...
        non_rally/
            frame_0001.jpg
            ...
    val/
        rally/
            ...
        non_rally/
            ...
"""

def build_model(num_classes=2, pretrained=True, freeze_backbone=True):
    # MobileNetV3 Small is extremely lightweight for live-stream analysis
    model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    # Replace the last classifier layer
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return model

def train_classifier(data_dir, num_epochs=8, batch_size=32, learning_rate=1e-3,
                     weight_decay=1e-4, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    # Class weights to counter imbalance: minority class gets a larger loss weight
    targets = np.array(image_datasets['train'].targets)
    class_counts = np.bincount(targets)
    class_weights = torch.tensor(class_counts.max() / class_counts, dtype=torch.float).to(device)
    print(f"Class counts: {class_counts.tolist()} | Class weights: {class_weights.tolist()}")

    model = build_model(num_classes=2, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.3)

    best_loss = float('inf')
    epochs_since_improvement = 0

    from tqdm import tqdm

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Wrap the dataloader with tqdm for a progress bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}", unit="batch", leave=False)

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar suffix with current batch loss
                pbar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best model based on validation loss + early stopping
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_since_improvement = 0
                    torch.save(model.state_dict(), "gatekeeper_best.pth")
                    print(f"  *** New best val loss: {best_loss:.4f} (saved) ***")
                else:
                    epochs_since_improvement += 1
                    print(f"  No improvement ({epochs_since_improvement}/{patience})")
                    if epochs_since_improvement >= patience:
                        print("Early stopping.")
                        return model

    return model

class RallyClassifier:
    def __init__(self, model_path, device=None, buffer_size=5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Debouncing Buffer
        self.buffer = deque(maxlen=buffer_size)

    def predict(self, frame):
        # frame: BGR image from OpenCV
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Use softmax to get probabilities (0.0 to 1.0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
            prediction = preds.item() # Assuming 0: non-rally, 1: rally
            confidence = conf.item()
            
        self.buffer.append(prediction)
        
        # Debouncing Logic: Return True only if majority of frames in buffer are 'Rally'
        is_rally_result = sum(self.buffer) > (len(self.buffer) / 2)
        
        return is_rally_result, confidence

def export_to_onnx(model_path, output_path="gatekeeper.onnx"):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=11)
    print(f"Model exported to {output_path}")

# Example Integration Loop
def analyze_video(video_source, model_path):
    classifier = RallyClassifier(model_path)
    cap = cv2.VideoCapture(video_source)

    frame_count = 0
    is_rally = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            is_rally = classifier.predict(frame)
            
        if is_rally:
            # FLAG ENABLED: Run downstream object detection (YOLO, etc.)
            # results = yolo_model(frame)
            # process_results(results)
            print(f"Frame {frame_count}: RALLY ACTIVE - Processing detections...")
        else:
            print(f"Frame {frame_count}: NON-RALLY - Skipping detection.")
            
        frame_count += 1
    
    cap.release()

if __name__ == "__main__":
    # To train:
    # train_classifier("path/to/dataset")

    # To export:
    # export_to_onnx("gatekeeper_best.pth")
    pass
