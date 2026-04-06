import torch
import torch.nn as nn
import time
import os

# ==========================================
# 1. DEFINE EXACT ARCHITECTURES USED
# ==========================================
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(38, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 53)

    def forward(self, x):
        return self.classifier(torch.relu(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))))

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(16, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, 53)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.classifier(x)

# ==========================================
# 2. PROFILING LOGIC
# ==========================================
def profile():
    print("--- Official Ninapro DB5 System Profile ---")
    
    teacher = TeacherNet()
    student = StudentNet()
    
    # Check if weights exist
    if os.path.exists("teacher_weights.pth"):
        teacher.load_state_dict(torch.load("teacher_weights.pth", weights_only=True))
    if os.path.exists("student_edge_model_v2.pth"):
        student.load_state_dict(torch.load("student_edge_model_v2.pth", weights_only=True))

    # --- A. PARAMETER COUNT (COMPRESSION) ---
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    compression = t_params / s_params
    
    print(f"\n[1] COMPRESSION METRICS")
    print(f"Teacher Parameters: {t_params:,}")
    print(f"Student Parameters: {s_params:,}")
    print(f"Final Compression Ratio: {compression:.2f}x")

    # --- B. INFERENCE LATENCY (AI THOUGHT SPEED) ---
    dummy_input = torch.randn(1, 16)
    student.eval()
    
    # Warmup
    for _ in range(100): _ = student(dummy_input)
    
    # Benchmark 1000 iterations
    start_time = time.perf_counter()
    for _ in range(1000):
        with torch.no_grad():
            _ = student(dummy_input)
    end_time = time.perf_counter()
    
    avg_inference_ms = ((end_time - start_time) / 1000) * 1000
    
    # --- C. SYSTEM LATENCY CALCULATION ---
    # Window stride is fixed at 50ms based on our preprocessing code
    stride_ms = 50.0 
    # Physics/IK overhead in PyBullet is approx 3-5ms
    physics_ms = 4.0 
    total_system_latency = stride_ms + avg_inference_ms + physics_ms

    print(f"\n[2] LATENCY METRICS")
    print(f"Avg AI Inference (CPU): {avg_inference_ms:.4f} ms")
    print(f"Data Collection Stride: {stride_ms} ms")
    print(f"Physics/IK Overhead:   {physics_ms} ms")
    print(f"------------------------------------")
    print(f"TOTAL SYSTEM LATENCY:  {total_system_latency:.3f} ms")
    
    if total_system_latency < 225:
        print("\nRESULT: SUCCESS. Below 225ms Cognitive Rejection Threshold.")

if __name__ == "__main__":
    profile()