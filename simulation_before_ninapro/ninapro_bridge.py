"""
ninapro_bridge_slow.py
Slow-Motion version for screen recording the presentation video.
"""
import pybullet as p
import pybullet_data
import time
import pandas as pd
import numpy as np
import os

# --- 1. MOCK DATA GENERATION ---
csv_filename = "mock_ninapro_50ms_stream.csv"
if not os.path.exists(csv_filename):
    print("Generating high-frequency Mock Ninapro DB5 data stream...")
    classes = [0]*100 + [1]*100 + [0]*100 
    data = []
    for i, cls in enumerate(classes):
        emg_channels = np.random.uniform(0.01, 0.1, 8) 
        if cls == 1: 
            emg_channels += np.random.uniform(0.5, 1.0, 8)
        data.append([i * 0.05] + list(emg_channels) + [cls])
    pd.DataFrame(data, columns=['timestamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'predicted_class']).to_csv(csv_filename, index=False)

# --- 2. INITIALIZE PYBULLET ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load Target Can
canId = p.createMultiBody(baseMass=0.3, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.12), 
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.12, rgbaColor=[1, 0, 0, 1]), 
                          basePosition=[0.5, 0, 0.06])
p.changeDynamics(canId, -1, lateralFriction=1.0)

# Load Robot
robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=1)
num_joints = p.getNumJoints(robotId)
end_effector_index = num_joints - 1 

# --- 3. THE SLOW-MOTION DATA BRIDGE ---
print("Starting Slow-Motion Presentation Stream...")
df = pd.read_csv(csv_filename)

pos_rest = [0.3, 0, 0.6]   
pos_approach = [0.5, 0, 0.2] 
pos_lift = [0.5, 0, 0.5]     

grasp_constraint = None 
current_target_pos = pos_rest
stride_time = 0.05 

for index, row in df.iterrows():
    current_intent = int(row['predicted_class'])
    
    # Intent Logic
    if current_intent == 1:
        current_target_pos = pos_approach if grasp_constraint is None else pos_lift
    else:
        current_target_pos = pos_rest
        if grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None

    # Calculate Inverse Kinematics
    joint_angles = p.calculateInverseKinematics(robotId, end_effector_index, current_target_pos)
    for j in range(num_joints):
        p.setJointMotorControl2(robotId, j, p.POSITION_CONTROL, targetPosition=joint_angles[j], force=500)

    # Physics Execution (SLOWED DOWN FOR VIDEO)
    steps_per_stride = int(stride_time * 240) 
    for _ in range(steps_per_stride):
        p.stepSimulation()
        
        # --- SLOW MOTION DELAY ---
        time.sleep(1./60.) # Forces it to render at 60fps visually instead of instantly
        
        # Check Grasp Proximity
        if current_target_pos == pos_approach and grasp_constraint is None:
            z_height = p.getLinkState(robotId, end_effector_index)[0][2]
            if z_height < 0.22: 
                grasp_constraint = p.createConstraint(robotId, end_effector_index, canId, -1, 
                                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
                current_target_pos = pos_lift 

print("\nSimulation Complete.")
p.disconnect()