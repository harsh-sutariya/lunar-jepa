import torch
from torch.utils.data import Dataset
import pandas as pd

class RajData(Dataset):
    def __init__(self, csv_file, device):

        self.device = device

        imu_colummns = ["id", "imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]
        pose_columns = ["id", "ground_truth_x", "ground_truth_y", "ground_truth_z", "ground_truth_roll", "ground_truth_pitch", "ground_truth_yaw"]
        current_velocity_columns = ["id", "ground_truth_v", "ground_truth_w"]

        self.data = pd.read_csv(csv_file)

        self.imu_data = self.data[imu_colummns].values
        self.pose_data = self.data[pose_columns].values
        self.current_velocity_data = self.data[current_velocity_columns].values

        self.imu_data = torch.tensor(self.imu_data, dtype=torch.float32, device=self.device)
        self.pose_data = torch.tensor(self.pose_data, dtype=torch.float32, device=self.device)
        self.current_velocity_data = torch.tensor(self.current_velocity_data, dtype=torch.float32, device=self.device)

        T=5
        imu_S = 200
        step_size = T

        self.imu_trajectories, self.delta_pose_trajectories, self.control_velocity_trajectories, self.trajectory_imu_ids, self.trajectory_pose_ids = self.preprocess_data(T, imu_S, step_size)
    
    def preprocess_data(self, T, imu_S=200, step_size=5):
        imu_ids = self.imu_data[:, 0]
        pose_ids = self.pose_data[:, 0]
        imu_values = self.imu_data[:, 1:]
        pose_values = self.pose_data[:, 1:]
        velocity_values = self.current_velocity_data[:, 1:]

        imu_trajectories = []
        delta_pose_trajectories = []
        control_velocity_trajectories = []
        trajectory_imu_ids = []
        trajectory_pose_ids = []
        
        for t in range(imu_S, len(self.imu_data) - T + 1, step_size):

            current_imu_ids = imu_ids[t-imu_S:t+T]
            current_pose_ids = pose_ids[t:t+T]
            if torch.all(current_imu_ids[imu_S:] == current_pose_ids):

                trajectory_imu = []
                for tau in range(T):
                    start_idx = t - imu_S + tau
                    end_idx = t + tau
                    sequence = imu_values[start_idx:end_idx]
                    trajectory_imu.append(sequence)
                
                trajectory_pose = []
                for tau in range(T):
                    current_pose = pose_values[t + tau]
                    previous_pose = pose_values[t + tau - 1] if tau > 0 else pose_values[t + tau]
                    delta_pose = current_pose - previous_pose
                    trajectory_pose.append(delta_pose)
                
                trajectory_velocity = []
                for tau in range(T - 1):
                    trajectory_velocity.append(velocity_values[t + tau + 1])
                
                trajectory_imu = torch.stack(trajectory_imu)
                trajectory_pose = torch.stack(trajectory_pose)
                trajectory_pose[0] = torch.zeros_like(trajectory_pose[0])
                trajectory_velocity = torch.stack(trajectory_velocity)
                
                imu_trajectories.append(trajectory_imu)
                delta_pose_trajectories.append(trajectory_pose)
                control_velocity_trajectories.append(trajectory_velocity)
                trajectory_imu_ids.append(current_imu_ids)
                trajectory_pose_ids.append(current_pose_ids)

        imu_trajectories = torch.stack(imu_trajectories)
        delta_pose_trajectories = torch.stack(delta_pose_trajectories)
        control_velocity_trajectories = torch.stack(control_velocity_trajectories)
        delta_pose_trajectories = delta_pose_trajectories.unsqueeze(2)
        control_velocity_trajectories = control_velocity_trajectories.unsqueeze(2)

        return imu_trajectories, delta_pose_trajectories, control_velocity_trajectories, trajectory_imu_ids, trajectory_pose_ids
    
    def __len__(self):
        return len(self.imu_trajectories)

    def __getitem__(self, idx):
        imu = self.imu_trajectories[idx]
        pose = self.delta_pose_trajectories[idx]
        velocity = self.control_velocity_trajectories[idx]
        
        return {
            'imu': imu,
            'pose': pose,
            'velocity': velocity,
            'imu_ids': self.trajectory_imu_ids[idx],
            'pose_ids': self.trajectory_pose_ids[idx]
        }

from torch.utils.data import random_split, DataLoader  

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RajData("./data.csv", device)
    total_length = len(dataset)
    train_length = int(0.7 * total_length)
    val_length = int(0.15 * total_length)
    test_length = total_length - train_length - val_length

    print(len(dataset))

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    idx = 2
    for batch in train_loader:
        imu = batch['imu'].to(device)
        pose = batch['pose'].to(device)
        velocity = batch['velocity'].to(device)
        while idx!=0:
            print(imu.shape, pose.shape, velocity.shape)
            print(idx)
            idx-=1
            