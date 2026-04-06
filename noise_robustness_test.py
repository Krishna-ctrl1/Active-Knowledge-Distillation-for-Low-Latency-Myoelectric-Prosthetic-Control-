"""
noise_robustness_test.py
RESEARCH TASK: Real-World Degradation Testing.
Injects Gaussian noise to simulate sweat, sensor shift, and electrical interference.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. REBUILD THE UNIFIED ARCHITECTURE
class NeuralNet(nn.Module):
    def __init__(self, num_classes=53):
        super(NeuralNet, self).__init__()
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

def add_gaussian_noise(data, noise_factor):
    """Injects mathematical static into the EMG signals"""
    std_dev = np.std(data, axis=0)
    # np.random.normal creates float64 (Double) by default
    noise = np.random.normal(0, std_dev, data.shape) * noise_factor
    return data + noise

def test_robustness():
    print("--- INITIATING SIGNAL DEGRADATION TEST ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    print("Loading Baseline and Distilled Models...")
    baseline_model = NeuralNet().to(device)
    baseline_model.load_state_dict(torch.load("baseline_no_distillation.pth", weights_only=True))
    baseline_model.eval()

    distilled_model = NeuralNet().to(device)
    distilled_model.load_state_dict(torch.load("student_edge_model_v2.pth", weights_only=True))
    distilled_model.eval()

    # Load a quick test sample (10,000 rows is enough for a statistical proof)
    print("Loading biological test data...")
    df = pd.read_csv("DB5_Master_Grasp_Dataset.csv")
    test_df = df.sample(n=10000, random_state=42)
    X_clean = test_df.iloc[:, 1:17].values.astype(np.float32)
    y_true = test_df.iloc[:, -1].values.astype(np.int64)
    y_tensor = torch.tensor(y_true).to(device)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    baseline_accs = []
    distilled_accs = []

    print("\n--- ATTACKING MODELS WITH NOISE ---")
    for noise in noise_levels:
        # 1. Corrupt the data
        X_noisy = add_gaussian_noise(X_clean, noise)
        
        # FIX: Force the noisy data back into 32-bit floats (Float)
        X_tensor = torch.tensor(X_noisy, dtype=torch.float32).to(device)
        
        # 2. Test Baseline
        with torch.no_grad():
            b_outputs = baseline_model(X_tensor)
            _, b_preds = torch.max(b_outputs, 1)
            b_acc = (b_preds == y_tensor).sum().item() / len(y_tensor) * 100
            baseline_accs.append(b_acc)
            
        # 3. Test Distilled
        with torch.no_grad():
            d_outputs = distilled_model(X_tensor)
            _, d_preds = torch.max(d_outputs, 1)
            d_acc = (d_preds == y_tensor).sum().item() / len(y_tensor) * 100
            distilled_accs.append(d_acc)
            
        print(f"Noise {int(noise*100):02d}% | Baseline Acc: {b_acc:.1f}% | Distilled Acc: {d_acc:.1f}%")

    # GENERATE THE PUBLICATION GRAPH
    print("\nRendering Robustness Curve...")
    plt.figure(figsize=(9, 6))
    plt.plot([n*100 for n in noise_levels], baseline_accs, marker='o', linestyle='dashed', color='red', label='Standard Baseline')
    plt.plot([n*100 for n in noise_levels], distilled_accs, marker='s', linewidth=2, color='blue', label='AKD Distilled Student')
    
    plt.title("Model Robustness Under Gaussian Noise Degradation", fontsize=14, fontweight='bold')
    plt.xlabel("Signal Noise Injection (%) - Simulating Sweat & Sensor Shift", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Noise_Robustness_Curve.png", dpi=300)
    print("[SUCCESS] Graph saved as 'Noise_Robustness_Curve.png'.")

if __name__ == "__main__":
    test_robustness()