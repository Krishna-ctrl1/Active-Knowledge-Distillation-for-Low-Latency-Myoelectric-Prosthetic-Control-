"""
subject_fine_tuning.py
RESEARCH TASK: Deep Personalized Calibration.
Personalizes the Global Model for a specific user.
Goal: Push accuracy past 90% using 50 epochs and a dynamic LR scheduler.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SubjectSpecificDataset(Dataset):
    def __init__(self, csv_file, subject_id):
        print(f"Extracting Deep Calibration Data for Subject {subject_id}...")
        df = pd.read_csv(csv_file)
        # Filter for only the selected subject
        subject_df = df[df['subject_id'] == subject_id]
        self.X = subject_df.iloc[:, 1:17].values.astype(np.float32)
        self.y = subject_df.iloc[:, -1].values.astype(np.int64)
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class StudentNet(nn.Module):
    def __init__(self, num_classes=53):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(16, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128); self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.classifier(x)

def fine_tune(target_subject=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    # 1. Load the Global Distilled Model (The Knowledge Base)
    model = StudentNet().to(device)
    model.load_state_dict(torch.load("student_edge_model_v2.pth", weights_only=True))
    
    # 2. Load data for JUST the Target Subject
    dataset = SubjectSpecificDataset("DB5_Master_Grasp_Dataset.csv", target_subject)
    train_size = int(0.7 * len(dataset)) # 70% for calibration, 30% for validation
    train_db, test_db = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_db, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=64, shuffle=False)
    
    # 3. Optimization Setup
    # Slightly higher starting LR since we have a scheduler to catch it
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    criterion = nn.CrossEntropyLoss()
    # Scheduler: Drops learning rate if accuracy stops improving for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print(f"\n--- COMMENCING EXTENDED CALIBRATION (Subject {target_subject}) ---")
    best_acc = 0.0
    
    for epoch in range(50): # Increased to 50 epochs
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward(); optimizer.step()
        
        # Subject-Specific Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, pred = torch.max(model(inputs), 1)
                total += labels.size(0); correct += (pred == labels).sum().item()
        
        acc = 100 * correct / total
        scheduler.step(acc)
        
        # Track the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/50 | LR: {current_lr:.6f} | Personalized Val Acc: {acc:.2f}%")
        
        # Save the absolute best weights
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"subject_{target_subject}_best_calibrated.pth")

    print(f"\n[RESULT] Deep Calibration Complete. PEAK Accuracy achieved: {best_acc:.2f}%")

if __name__ == "__main__":
    fine_tune(target_subject=1)