import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import wandb
import pandas as pd

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

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
                # Process IMU data
                trajectory_imu = []
                for tau in range(T):
                    start_idx = t - imu_S + tau
                    end_idx = t + tau
                    sequence = imu_values[start_idx:end_idx]
                    trajectory_imu.append(sequence)
                
                # Process current delta poses
                trajectory_pose = []
                for tau in range(T):
                    current_pose = pose_values[t + tau]
                    previous_pose = pose_values[t + tau - 1] if tau > 0 else pose_values[t + tau]
                    delta_pose = current_pose - previous_pose
                    trajectory_pose.append(delta_pose)
                
                # Process velocity data
                trajectory_velocity = []
                for tau in range(T - 1):
                    trajectory_velocity.append(velocity_values[t + tau + 1])
                
                # Stack and process tensors
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
        
        # Add dimension for consistency
        delta_pose_trajectories = delta_pose_trajectories.unsqueeze(2)
        control_velocity_trajectories = control_velocity_trajectories.unsqueeze(2)

        return imu_trajectories, delta_pose_trajectories, control_velocity_trajectories, trajectory_imu_ids, trajectory_pose_ids
    
    def __len__(self):
        return len(self.imu_trajectories)

    def __getitem__(self, idx):
        imu = self.imu_trajectories[idx]
        velocity = self.control_velocity_trajectories[idx]
        pose = self.delta_pose_trajectories[idx]
        
        return {
            'imu': imu,
            'pose': pose,
            'velocity': velocity,
            'imu_ids': self.trajectory_imu_ids[idx],
            'pose_ids': self.trajectory_pose_ids[idx]
        }


class PoseProber(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 6)
        )
    
    def forward(self, x):
        return self.network(x)


def train_pose_prober(jepa, train_loader, val_loader, device, 
                     num_epochs=300, lr=1e-3, weight_decay=1e-5):

    input_dim = jepa.repr_dim
    prober = PoseProber(input_dim).to(device)
    optimizer = torch.optim.Adam(prober.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    wandb.watch(prober, log="all", log_freq=10)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        prober.train()
        jepa.eval()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            with torch.no_grad():

                imu = batch['imu'].to(device)
                delta_pose = batch['pose'].to(device)
                control_velocity = batch['velocity'].to(device)
                
                predicted_states = jepa(imu=imu, delta_pose=delta_pose, 
                                     control_velocity=control_velocity)
                
            pred_poses = prober(predicted_states)
            pred_poses = pred_poses.unsqueeze(2)
            
            loss = criterion(pred_poses, delta_pose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        prober.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imu = batch['imu'].to(device)
                delta_pose = batch['pose'].to(device)
                control_velocity = batch['velocity'].to(device)

                predicted_states = jepa(imu=imu, delta_pose=delta_pose, 
                                     control_velocity=control_velocity)
                
                pred_poses = prober(predicted_states)
                pred_poses = pred_poses.unsqueeze(2)
                
                loss = criterion(pred_poses, delta_pose)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(prober.state_dict(), 'pose_prober_gpu.pth')
        
        scheduler.step(val_loss)
    
    return prober

if __name__ == "__main__":
    device = get_device()
    
    jepa = models.JEPA(device=device)
    jepa_path = './jepa/1737687243.3784735/64_0.001_best_model.pth'
    jepa.load_state_dict(torch.load(jepa_path))
    jepa.eval()

    data = RajData("./data.csv", device)
    total_length = len(data)
    train_length = int(0.7 * total_length)
    val_length = int(0.15 * total_length)
    test_length = total_length - train_length - val_length
    train_dataset, val_dataset, test_dataset = random_split(data, [train_length, val_length, test_length])

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    wandb.init(project="pose_prober", config={
        "learning_rate": 1e-3,
        "epochs": 300,
        "batch_size": 32,
        "hidden_dim": 256,
        "model_path": jepa_path
    })
    
    # Train the prober
    pose_prober = train_pose_prober(jepa, train_loader, val_loader, device)

    wandb.finish()
