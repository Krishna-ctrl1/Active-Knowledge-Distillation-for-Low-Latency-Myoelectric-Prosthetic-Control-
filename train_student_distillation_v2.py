"""
train_student_distillation_v2.py
Upgraded Architecture for >85% Accuracy KPI.
Introduces BatchNorm, Adaptive Learning Rate, and Teacher-Heavy Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ==========================================
# 1. DATASET
# ==========================================
class NinaproDistillationDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading Master Dataset for Distillation...")
        df = pd.read_csv(csv_file)
        
        # Student Input: 16 EMG channels
        self.emg_only = df.iloc[:, 1:17].values.astype(np.float32)
        
        # Teacher Input: 38 Features (16 EMG + 22 Glove)
        self.all_features = df.iloc[:, 1:-1].values.astype(np.float32)
        
        # Labels
        self.y = df.iloc[:, -1].values.astype(np.int64)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.emg_only[idx], self.all_features[idx], self.y[idx]

# ==========================================
# 2. UPGRADED ARCHITECTURES
# ==========================================
class TeacherNet(nn.Module):
    def __init__(self, num_classes=53):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(38, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.classifier(x)

class StudentNet(nn.Module):
    def __init__(self, num_classes=53):
        super(StudentNet, self).__init__()
        # Upgraded Capacity with Batch Normalization
        self.fc1 = nn.Linear(16, 256)
        self.bn1 = nn.BatchNorm1d(256) # Stabilizes EMG variance
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128) # Accelerates convergence
        
        self.classifier = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.classifier(x)

# ==========================================
# 3. TEACHER-HEAVY DISTILLATION LOSS
# ==========================================
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=4.0, alpha=0.2):
    # alpha=0.2: 80% learning from Teacher, 20% from hard labels
    hard_loss = F.cross_entropy(student_logits, true_labels)
    
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return (alpha * hard_loss) + ((1. - alpha) * soft_loss)

# ==========================================
# 4. TRAINING LOOP WITH ADAPTIVE SCHEDULER
# ==========================================
def train_student(csv_path="DB5_Master_Grasp_Dataset.csv"):
    dataset = NinaproDistillationDataset(csv_path)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Load Frozen Teacher
    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load("teacher_weights.pth", weights_only=True))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Initialize Student and Optimizer
    student = StudentNet().to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.002) # Slightly higher starting LR
    
    # Adaptive Scheduler: Reduces LR by half if validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    epochs = 75 
    print("\nStarting Upgraded Active Distillation (Target: >85%)...")
    
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        
        for emg_inputs, all_inputs, labels in train_loader:
            emg_inputs, all_inputs, labels = emg_inputs.to(device), all_inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher(all_inputs)
                
            student_logits = student(emg_inputs)
            loss = distillation_loss(student_logits, teacher_logits, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation Phase
        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for emg_inputs, _, labels in test_loader:
                emg_inputs, labels = emg_inputs.to(device), labels.to(device)
                outputs = student(emg_inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        # Step the scheduler based on the validation accuracy
        if isinstance(optimizer.param_groups[0]['lr'], float): # Avoids deprecation warning formatting issues
            current_lr = optimizer.param_groups[0]['lr']
        else:
             current_lr = optimizer.param_groups[0]['lr'].item()
             
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Student Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

    print("\nSaving Upgraded Student Weights...")
    torch.save(student.state_dict(), "student_edge_model_v2.pth")
    print("Success! Edge-ready Student Model saved as 'student_edge_model_v2.pth'")

if __name__ == "__main__":
    train_student()