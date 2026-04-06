"""
train_teacher.py
Trains the heavy Multimodal Teacher Network (16 sEMG + 22 Kinematic Glove features).
Goal: >90% Accuracy to act as the Ground Truth for distillation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class NinaproDB5Dataset(Dataset):
    def __init__(self, csv_file):
        print("Loading Master Dataset into Memory...")
        df = pd.read_csv(csv_file)
        
        # We drop the subject_id for this global training phase
        # Features are columns 1 to 38 (16 EMG + 22 Glove)
        self.X = df.iloc[:, 1:-1].values.astype(np.float32)
        
        # Labels are the last column
        self.y = df.iloc[:, -1].values.astype(np.int64)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. THE TEACHER NETWORK ARCHITECTURE
# ==========================================
class TeacherNet(nn.Module):
    def __init__(self, num_classes=53): # DB5 has classes 0-52
        super(TeacherNet, self).__init__()
        
        # The Teacher takes ALL 38 features (16 EMG + 22 Glove)
        self.fc1 = nn.Linear(38, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Prevents overfitting

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.classifier(x)

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_teacher(csv_path="DB5_Master_Grasp_Dataset.csv"):
    # Load Data
    full_dataset = NinaproDB5Dataset(csv_path)
    
    # Split into 80% Training and 20% Testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on device: {device}")
    
    model = TeacherNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    
    print("\nStarting Teacher Training Phase...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("\nSaving Teacher Weights...")
    torch.save(model.state_dict(), "teacher_weights.pth")
    print("Success! Teacher Model saved as 'teacher_weights.pth'")

if __name__ == "__main__":
    train_teacher()