# IMU-JEPA: Joint Embedding Predictive Architecture for IMU Data

This repository contains the implementation of a Joint Embedding Predictive Architecture (JEPA) for processing Inertial Measurement Unit (IMU), Pose and command velocity data. The model is designed to learn meaningful representations from IMU sensor data that can be used for various downstream tasks such as pose estimation and localization. We can also use these representations to make a planning architecture using MPC to get control velocities as actions to reach a particular place in the environment.

## Repository Structure

- `train.py`: Main training script for the JEPA model
- `models.py`: Contains model architectures and implementations
- `data_structure.py`: Data handling and preprocessing utilities
- `train_pose_prober.py`: Training script for the pose estimation prober
- `train_localization_prober.py`: Training script for the localization prober
- `test_localization_prober.py`: Testing script for the localization prober

## Recurrent JEPA Model Architecture
![lunar-jepa drawio](https://github.com/user-attachments/assets/baa77564-0a4b-4424-8782-f7c35b5e4b4e)


## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Weights & Biases (wandb) for experiment tracking
- Matplotlib
- TQDM

### Training Pipeline

To replicate the results, follow these steps:

1. First, train the main JEPA model:
```bash
python train.py
```

2. After the JEPA model is trained, train the probers for downstream tasks:
```bash
python train_pose_prober.py
python train_localization_prober.py
```

3. You can evaluate the localization performance using:
```bash
python test_localization_prober.py
```

## Training Logs

For detailed training logs and experiment results of the JEPA model, please refer to the image below:
![image](https://github.com/user-attachments/assets/1a2e94ef-30f7-4c72-abfa-6f851cde71cc)


## Dataset

The repository uses IMU, Pose and Velocity data obtained from Carla simulator's Lunar environment. The data is stored in `data.csv` (64MB). Make sure this file is present in the root directory before starting the training process. This data is then divided into trajectories, for more information follow `data_structure.py`.

## References

1. Lunar Autonomy Challenge https://lunar-autonomy-challenge.jhuapl.edu/
2. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning https://arxiv.org/abs/2411.04983
3. Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models https://latent-planning.github.io/static/paper.pdf
4. Psuedo-Recurrent JEPA for two-room environment https://youtu.be/SOw40mduJ1o
5. A Path Towards Autonomous Machine Intelligence https://openreview.net/pdf?id=BZ5a1r-kVsf
