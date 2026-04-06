"""
train_student_distillation.py
Executes the Active Knowledge Distillation. 
Trains the lightweight Student (16 sEMG) using the heavy Teacher (16 sEMG + 22 Glove).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ==========================================
# 1. DISTILLATION DATASET (Splitting the inputs)
# ==========================================
class NinaproDistillationDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading Master Dataset for Distillation...")
        df = pd.read_csv(csv_file)
        
        # Student Input: ONLY the 16 EMG channels (Columns 1 to 16)
        self.emg_only = df.iloc[:, 1:17].values.astype(np.float32)
        
        # Teacher Input: ALL 38 Features (16 EMG + 22 Glove)
        self.all_features = df.iloc[:, 1:-1].values.astype(np.float32)
        
        # Target Labels
        self.y = df.iloc[:, -1].values.astype(np.int64)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.emg_only[idx], self.all_features[idx], self.y[idx]

# ==========================================
# 2. NETWORK ARCHITECTURES
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
        # Extremely lightweight - Input is only 16!
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.classifier(x)

# ==========================================
# 3. ACTIVE DISTILLATION LOSS
# ==========================================
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    """
    alpha: 0.5 means it learns 50% from the hard labels and 50% from the Teacher's soft labels.
    temperature: 3.0 softens the Teacher's probabilities to reveal the 'hidden context' of the grasp.
    """
    hard_loss = F.cross_entropy(student_logits, true_labels)
    
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return (alpha * hard_loss) + ((1. - alpha) * soft_loss)

# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
def train_student(csv_path="DB5_Master_Grasp_Dataset.csv"):
    dataset = NinaproDistillationDataset(csv_path)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Teacher and Freeze It
    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load("teacher_weights.pth", weights_only=True))
    teacher.eval() # Set to evaluation mode
    for param in teacher.parameters():
        param.requires_grad = False # Freeze weights
        
    print("Pre-trained Teacher loaded and frozen.")
    
    # Initialize Student
    student = StudentNet().to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    # ADD THIS SCHEDULER: Drops learning rate by 50% every 15 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5) 
    
    epochs = 50 # INCREASE EPOCHS
    print("\nStarting Active Distillation Phase...")
    
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for emg_inputs, all_inputs, labels in train_loader:
            emg_inputs, all_inputs, labels = emg_inputs.to(device), all_inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 1. Get Teacher's predictions (No gradients needed)
            with torch.no_grad():
                teacher_logits = teacher(all_inputs)
                
            # 2. Get Student's predictions
            student_logits = student(emg_inputs)
            
            # 3. Calculate Distillation Loss
            loss = distillation_loss(student_logits, teacher_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation Phase
        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for emg_inputs, _, labels in test_loader: # Student doesn't get 'all_inputs' here!
                emg_inputs, labels = emg_inputs.to(device), labels.to(device)
                outputs = student(emg_inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Student Train Acc: {train_acc:.2f}% | Student Val Acc: {val_acc:.2f}%")

    print("\nSaving Distilled Student Weights...")
    torch.save(student.state_dict(), "student_edge_model.pth")
    print("Success! Edge-ready Student Model saved as 'student_edge_model.pth'")

if __name__ == "__main__":
    train_student()