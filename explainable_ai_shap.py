"""
explainable_ai_shap.py
RESEARCH TASK: Explainable AI (XAI) via SHAP.
Generates a Feature Importance Heatmap to prove biological validity.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

class StudentNet(nn.Module):
    def __init__(self, num_classes=53):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(16, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.classifier(x)

def generate_shap_analysis(subject_id=1):
    print("--- INITIATING SHAP EXPLAINABLE AI ANALYSIS ---")
    device = torch.device("cpu") 
    
    model = StudentNet().to(device)
    model.load_state_dict(torch.load(f"subject_{subject_id}_best_calibrated.pth", weights_only=True))
    model.eval()

    print("Loading biological data for baseline comparison...")
    df = pd.read_csv("DB5_Master_Grasp_Dataset.csv")
    subject_df = df[df['subject_id'] == subject_id]
    X_raw = subject_df.iloc[:, 1:17].values.astype(np.float32)
    
    background_data = torch.tensor(X_raw[np.random.choice(X_raw.shape[0], 500, replace=False)])
    test_data = torch.tensor(X_raw[np.random.choice(X_raw.shape[0], 100, replace=False)])

    print("Dissecting Neural Network mathematically. This may take a minute...")
    try:
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(test_data)
    except Exception:
        print("Using GradientExplainer for robust gradient tracking...")
        explainer = shap.GradientExplainer(model, background_data)
        shap_values = explainer.shap_values(test_data)

    print("Condensing 53-class multi-dimensional SHAP values...")
    # FIX: Condense the 3D multi-class array into a 2D global importance array
    if isinstance(shap_values, list):
        # Sum the absolute SHAP values across all 53 classes
        shap_values_2d = np.sum(np.abs(shap_values), axis=0)
    else:
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            # Identify the class dimension (53) and sum across it
            class_dim = shap_array.shape.index(53) if 53 in shap_array.shape else 1
            shap_values_2d = np.sum(np.abs(shap_array), axis=class_dim)
        else:
            shap_values_2d = shap_array

    print("Rendering Feature Importance Chart...")
    feature_names = [f"EMG Ch {i+1}" for i in range(16)]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Global SHAP Feature Importance: Calibrated Model", fontsize=14, fontweight='bold')
    
    # Pass the condensed 2D array to the plotter
    shap.summary_plot(shap_values_2d, test_data.numpy(), feature_names=feature_names, show=False)
    
    plt.tight_layout()
    plt.savefig("SHAP_Feature_Importance.png", dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Graph saved as 'SHAP_Feature_Importance.png'.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') 
    generate_shap_analysis(subject_id=1)