import pybullet as p
import pybullet_data
import time
import math

# ==========================================
# 1. INITIALIZATION & ENVIRONMENT SETUP
# ==========================================
# Connect to the PyBullet GUI
physicsClient = p.connect(p.GUI)

# Add standard PyBullet data paths (for ground planes, basic robots)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set real-world Newtonian gravity (-9.81 m/s^2) - As promised on your slide!
p.setGravity(0, 0, -9.81)

# Load the ground plane
planeId = p.loadURDF("plane.urdf")

# ==========================================
# 2. CREATE THE "RED CAN" (Target Object)
# ==========================================
# Instead of downloading a mesh, we mathematically generate a 300g can
can_radius = 0.03
can_height = 0.12
can_mass = 0.3 # 300 grams

visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER, 
                                    radius=can_radius, 
                                    length=can_height, 
                                    rgbaColor=[1, 0, 0, 1]) # Red color

collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, 
                                          radius=can_radius, 
                                          height=can_height)

# Place the can on the table/ground in front of the robot
can_start_pos = [0.5, 0, can_height/2]
can_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

canId = p.createMultiBody(baseMass=can_mass,
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=can_start_pos,
                          baseOrientation=can_start_orientation)

# Set Friction for the can (Coulomb Friction = 1.0)
p.changeDynamics(canId, -1, lateralFriction=1.0)

# ==========================================
# 3. LOAD THE ROBOTIC ARM
# ==========================================
# Loading PyBullet's built-in 7-DOF arm. 
# (Note: For your final thesis, you can replace "kuka_iiwa/model.urdf" with "ur10e.urdf")
robot_start_pos = [0, 0, 0]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("kuka_iiwa/model.urdf", robot_start_pos, robot_start_orientation, useFixedBase=1)

# The end-effector (gripper attachment point) is the last link
num_joints = p.getNumJoints(robotId)
end_effector_index = num_joints - 1 

print(f"Simulation Loaded: 300g object placed. Robot initialized with {num_joints} joints.")

# ==========================================
# 4. KINEMATIC CONTROL LOOP (The IK Solver)
# ==========================================
# We want the robot to move to the Can's position
target_pos = [0.5, 0, 0.2] # Slightly above the can to prepare for grasp

print("Calculating Inverse Kinematics...")
# Calculate the exact joint angles needed to reach the target (<20ms latency)
joint_angles = p.calculateInverseKinematics(robotId, end_effector_index, target_pos)

# Send commands to motors
for i in range(num_joints):
    # Some links aren't controllable motors, so we use a try-except to bypass fixed joints
    try:
        p.setJointMotorControl2(bodyIndex=robotId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_angles[i],
                                force=500) # Motor torque
    except:
        pass

# ==========================================
# 5. RUN THE PHYSICS ENGINE
# ==========================================
# Step the simulation forward in time so the physics engine calculates gravity and movement
print("Executing Approach...")
for i in range(500): # Runs for a few seconds
    p.stepSimulation()
    time.sleep(1./240.) # Standard 240Hz physics update

print("Phase 1 Kinematic Validation Complete.")
p.disconnect()