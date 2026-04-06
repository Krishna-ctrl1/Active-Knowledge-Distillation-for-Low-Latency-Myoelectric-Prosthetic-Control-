"""
real_sim_pipeline.py
The Ultimate Presentation Demo.
Drives the PyBullet physics simulation using the REAL trained PyTorch Student model.
"""
from collections import deque
import torch
import torch.nn as nn
import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. LOAD THE TRAINED AI (The Upgraded Brain)
# ==========================================
class StudentNet(nn.Module):
    def __init__(self, num_classes=53):
        super(StudentNet, self).__init__()
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

device = torch.device("cpu") 
ai_model = StudentNet().to(device)
ai_model.load_state_dict(torch.load("student_edge_model_v2.pth", weights_only=True))
ai_model.eval()
print("Upgraded AI Brain Loaded Successfully.")

# ==========================================
# 2. LOAD REAL HUMAN DATA
# ==========================================
print("Loading Real DB5 Human EMG Data...")
df = pd.read_csv("DB5_Master_Grasp_Dataset.csv")

# Grab a slice of continuous real-time data transitioning from Rest to Grasp
demo_data = df.iloc[500:900] 

# ==========================================
# 3. INITIALIZE DIGITAL TWIN
# ==========================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

canId = p.createMultiBody(baseMass=0.3, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.12), 
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.12, rgbaColor=[1, 0, 0, 1]), 
                          basePosition=[0.5, 0, 0.06])
p.changeDynamics(canId, -1, lateralFriction=1.0)

robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=1)
num_joints = p.getNumJoints(robotId)
end_effector_index = num_joints - 1 

pos_rest = [0.3, 0, 0.6]   
pos_approach = [0.5, 0, 0.2] 
pos_lift = [0.5, 0, 0.5]     

grasp_constraint = None 
current_target_pos = pos_rest

# ==========================================
# 4. REAL-TIME INFERENCE LOOP
# ==========================================
print("\n=== STARTING CLOSED-LOOP INFERENCE ===")
time.sleep(2) 
# 250ms smoothing buffer (5 frames * 50ms stride)
prediction_buffer = deque(maxlen=5)
for index, row in demo_data.iterrows():
    emg_raw = row.iloc[1:17].values.astype(np.float32)
    true_intent = int(row['gesture_class'])
    
    # AI INFERENCE
    emg_tensor = torch.tensor(emg_raw).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction_logits = ai_model(emg_tensor)
        raw_intent = torch.argmax(prediction_logits, dim=1).item()
        
    # SMOOTHING FILTER (The Majority Vote)
    prediction_buffer.append(raw_intent)
    predicted_intent = max(set(prediction_buffer), key=prediction_buffer.count)
        
    # INTENT TRANSLATION
    if predicted_intent != 0:
        action = "GRASPING"
        current_target_pos = pos_approach if grasp_constraint is None else pos_lift
    else:
        action = "RESTING"
        current_target_pos = pos_rest
        if grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
        
    # KINEMATIC ACTUATION
    joint_angles = p.calculateInverseKinematics(robotId, end_effector_index, current_target_pos)
    for j in range(num_joints):
        p.setJointMotorControl2(robotId, j, p.POSITION_CONTROL, targetPosition=joint_angles[j], force=500)

    # PHYSICS ENGINE
    for _ in range(12): 
        p.stepSimulation()
        time.sleep(1./60.) 
        
        if current_target_pos == pos_approach and grasp_constraint is None:
            if p.getLinkState(robotId, end_effector_index)[0][2] < 0.22: 
                grasp_constraint = p.createConstraint(robotId, end_effector_index, canId, -1, 
                                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
                current_target_pos = pos_lift 

    match = "✅" if predicted_intent == true_intent else "❌"
    print(f"Human True Intent: {true_intent:2d} | AI Prediction: {predicted_intent:2d} {match} | Robot Action: {action}")

print("\n=== DEMO COMPLETE ===")
p.disconnect()