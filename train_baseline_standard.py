"""
train_baseline_standard.py
RESEARCH TASK: Baseline Ablation Study.
Trains a 16-channel EMG model WITHOUT Teacher Distillation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 1. DATASET
class StandardDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading Dataset for Baseline Study...")
        df = pd.read_csv(csv_file)
        # We only take the 16 EMG channels (Columns 1 to 16)
        self.X = df.iloc[:, 1:17].values.astype(np.float32) 
        self.y = df.iloc[:, -1].values.astype(np.int64)    
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# 2. ARCHITECTURE (Must be identical to the Student for a fair test)
class BaselineNet(nn.Module):
    def __init__(self, num_classes=53):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(16, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.classifier(x)

# 3. TRAINING LOOP
def train_baseline():
    dataset = StandardDataset("DB5_Master_Grasp_Dataset.csv")
    train_size = int(0.8 * len(dataset))
    train_db, test_db = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_db, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    model = BaselineNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    print("\n--- STARTING BASELINE TRAINING (NO DISTILLATION) ---")
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/30 | Loss: {running_loss/len(train_loader):.4f} | Baseline Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "baseline_no_distillation.pth")
    print("\n[RESULT] Baseline model saved. Compare this to your 82.44% Distilled Model!")

if __name__ == "__main__":
    train_baseline()