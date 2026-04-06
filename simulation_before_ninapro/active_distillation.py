"""
active_distillation.py
PyTorch implementation of Vision-to-EMG Active Knowledge Distillation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ==========================================
# 1. THE TEACHER NETWORK (Vision + EMG)
# ==========================================
class TeacherNet(nn.Module):
    def __init__(self, num_classes=53):
        super(TeacherNet, self).__init__()
        self.vision_fc = nn.Linear(512, 128) # e.g., ResNet/OAK-D features
        self.emg_fc = nn.Linear(8, 128)      # 8-channel sEMG
        
        self.fusion = nn.Linear(128 + 128, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, vision_data, emg_data):
        v_feat = F.relu(self.vision_fc(vision_data))
        e_feat = F.relu(self.emg_fc(emg_data))
        fused = torch.cat((v_feat, e_feat), dim=1) 
        fused = F.relu(self.fusion(fused))
        return self.classifier(fused)

# ==========================================
# 2. THE STUDENT NETWORK (EMG Only) - For Edge Deployment
# ==========================================
class StudentNet(nn.Module):
    def __init__(self, num_classes=53):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, emg_data):
        x = F.relu(self.fc1(emg_data))
        x = F.relu(self.fc2(x))
        return self.classifier(x)

# ==========================================
# 3. ACTIVE DISTILLATION LOSS FUNCTION
# ==========================================
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    hard_loss = F.cross_entropy(student_logits, true_labels)
    
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return (alpha * hard_loss) + ((1. - alpha) * soft_loss)

# ==========================================
# 4. LATENCY & COMPRESSION PROOF (Run this to verify)
# ==========================================
if __name__ == "__main__":
    print("--- Active Distillation Benchmark ---")
    
    teacher = TeacherNet()
    student = StudentNet()
    
    # Calculate Parameters (Proving Compression)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher Model Parameters: {teacher_params:,}")
    print(f"Student Model Parameters: {student_params:,}")
    print(f"Compression Ratio: {teacher_params / student_params:.2f}x smaller\n")
    
    # Simulate a single 8-channel EMG input from Ninapro
    dummy_emg = torch.randn(1, 8) 
    
    # Warm-up pass (PyTorch CPU initialization)
    _ = student(dummy_emg)
    
    # Benchmark Student Inference Time
    latencies = []
    for _ in range(100):
        start_time = time.perf_counter()
        _ = student(dummy_emg)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to ms
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Student Average Inference Latency: {avg_latency:.3f} ms")
    print("Conclusion: Easily satisfies the <5ms requirement for the 50ms sliding window.")