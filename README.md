# Active Knowledge Distillation for Low-Latency Myoelectric Prosthetic Control

This repository houses an end-to-end PyTorch-based framework designed for the rapid and robust translation of Electromyography (EMG) signals into intent for myoelectric prostheses. The primary objective is to achieve highly accurate, low-latency control suitable for real-time edge devices using **Active Knowledge Distillation (AKD)**.

The project demonstrates that a streamlined "Student" model, distilled from a complex "Teacher" model, can achieve parity in gesture classification capability while significantly reducing inference latency, paving the way for instantaneous prosthetic response.

## 🚀 Key Features

*   **Knowledge Distillation Framework:** Implements Teacher computing structures and Distilled Student Edge models for efficient on-device processing.
*   **Leave-One-Subject-Out (LOSO) Validation:** Robust evaluation mimicking real-world blind testing on unseen human subjects.
*   **Subject-Specific Fine-Tuning:** Scripts for rapid calibration and fine-tuning to specific individuals.
*   **Real-Time Simulation Digital Twin:** A PyBullet-driven simulation that demonstrates closed-loop control of a robotic arm (Kuka iiwa) based on real-time continuous EMG data prediction.
*   **Noise Robustness Profiling:** Integrated tests simulating real-world signal degradation and sensor noise.
*   **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) methodology combined to interpret model attention and feature importance.
*   **NinaPro DB5 Integration:** Pre-configured data ingestion for the NinaPro Database 5 dataset.

## 📁 Repository Structure

*   **Training & Distillation**
    *   `train_teacher.py` - Trains the complex, high-capacity baseline model.
    *   `train_student_distillation.py` / `_v2` - Performs knowledge distillation to train the lightweight edge student models.
    *   `train_baseline_standard.py` - Trains a student architecture without distillation for baseline comparisons.
*   **Evaluation & Adaptation**
    *   `train_loso_validation.py` - Validates model generalization over unseen subjects.
    *   `subject_fine_tuning.py` - Custom calibration script for optimizing models to individual users.
    *   `noise_robustness_test.py` - Evaluates the model against synthetic noise (Gaussian, dropout, signal drift).
    *   `explainable_ai_shap.py` - Generates visual explanations for model feature attributions.
    *   `profile_system.py` - Latency and throughput benchmarking for edge deployment.
*   **Simulation & Demo**
    *   `real_sim_pipeline.py` - The ultimate presentation demo. Drives a PyBullet physics simulation using the distilled model on a continuous EMG stream.
*   **Data Prep**
    *   `ninapro_preprocess.py` - Scripts for filtering, smoothing, and windowing the massive DB5 raw data.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Krishna-ctrl1/Active-Knowledge-Distillation-for-Low-Latency-Myoelectric-Prosthetic-Control-.git
    cd Active-Knowledge-Distillation-for-Low-Latency-Myoelectric-Prosthetic-Control-
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `.\env\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If installing PyBullet fails, ensure you have the appropriate C++ build tools installed for your operating system.*

## 📖 Usage

### 1. Data Preparation
Ensure the DB5 Dataset (`DB5_Master_Grasp_Dataset.csv`) is located in the root directory before running any models.

### 2. Training Pipeline
To retrain the models from scratch:
1. Train the Teacher: `python train_teacher.py`
2. Train the Student via Distillation: `python train_student_distillation_v2.py`
3. Optional: Fine tune on a specific subject: `python subject_fine_tuning.py`

### 3. Running the Simulation
To view the PyBullet digital twin in action using the pre-trained `student_edge_model_v2.pth`:
```bash
python real_sim_pipeline.py
```
This script will initialize a robotic arm, load real unseen human data, process predictions through a smoothing filter (Majority Vote), and translate those predictions into physical kinematics in real-time.

### 4. Explainable AI & Profiling
To understand which EMG channels the model weighs most heavily, run:
```bash
python explainable_ai_shap.py
```
To benchmark model latency against your hardware:
```bash
python profile_system.py
```

## 🧠 Core Methodology

The transition from a resting position to a target pose (e.g., Grasping) relies on a continuous 50ms stride inference loop. The framework ensures:
1. **Low Latency:** The student edge model (`StudentNet`) relies on optimized Batch Normalized Linear layers requiring vastly fewer FLOPs than convolutions or RNNs.
2. **Signal Stability:** A predictive 250ms smoothing buffer effectively filters out transient noise without introducing noticeable mechanical lag.

## 🤝 Contribution
Contributions to improve noise robustness logic, refine the inverse kinematics calculation in the simulator, or extend the framework to new datasets are welcome.
