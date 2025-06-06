# Visual-Inertial Odometry (VIO) Fusion with LSTM

This project implements a deep learning-based Visual-Inertial Odometry (VIO) system using PyTorch. It includes three model types: Vision-only, Inertial-only, and Fusion-based models, each using LSTM networks to estimate poses from simulated or real sensor data.

## 🧠 Model Types

1. **VisionOnlyModel**: LSTM model processing vision features.
2. **InertialOnlyModel**: LSTM model processing IMU sequences.
3. **FusionModel**: Combines both vision and inertial inputs through dedicated LSTMs and concatenates their outputs for pose estimation.

## 📁 File Structure

- `vio.py` – Core model definitions for VIO using PyTorch.

## 🛠️ Dependencies

- Python 3.8+
- PyTorch
- NumPy

Install dependencies using:

```bash
pip install torch numpy
