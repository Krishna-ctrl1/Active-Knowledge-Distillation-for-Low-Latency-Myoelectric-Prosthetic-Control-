"""
train_loso_validation.py
RESEARCH TASK: Leave-One-Subject-Out (LOSO) Cross-Validation.
Trains on Subjects 1-9, Tests strictly on Subject 10.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class LOSODataset(Dataset):
    def __init__(self, csv_file, exclude_subject=None, only_subject=None):
        print("Loading DB5 Dataset for LOSO...")
        df = pd.read_csv(csv_file)
        
        if exclude_subject is not None:
            print(f"Filtering: Training on Everyone EXCEPT Subject {exclude_subject}")
            df = df[df['subject_id'] != exclude_subject]
        elif only_subject is not None:
            print(f"Filtering: Testing ONLY on Subject {only_subject}")
            df = df[df['subject_id'] == only_subject]
            
        self.X = df.iloc[:, 1:17].values.astype(np.float32) 
        self.y = df.iloc[:, -1].values.astype(np.int64)    
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class BaselineNet(nn.Module):
    def __init__(self, num_classes=53):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(16, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128); self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.classifier(x)

def train_loso(test_subject=10):
    train_db = LOSODataset("DB5_Master_Grasp_Dataset.csv", exclude_subject=test_subject)
    test_db = LOSODataset("DB5_Master_Grasp_Dataset.csv", only_subject=test_subject)
    
    train_loader = DataLoader(train_db, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print(f"\n--- STARTING LOSO TRAINING (TESTING ON SUBJ {test_subject}) ---")
    for epoch in range(15): # 15 epochs is enough to see the collapse
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward(); optimizer.step()
        
        # Validation on Unseen Subject
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs), 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/15 | LOSO Unseen Subject Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    train_loso(test_subject=10)