import models
import data_structure
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

class LocalizationProberData(Dataset):
    def __init__(self, predicted_states, ground_truth_poses, device):
        self.device = device
        self.predicted_states = predicted_states.to(device)  # Shape: (B, T, D)
        self.ground_truth_poses = ground_truth_poses.to(device)  # Shape: (B, T, 6)
        
        # Compute delta poses
        # self.delta_poses = torch.zeros_like(self.ground_truth_poses)
        # self.delta_poses[:, 1:] = self.ground_truth_poses[:, 1:] - self.ground_truth_poses[:, :-1]
    
    def __len__(self):
        return len(self.predicted_states)
    
    def __getitem__(self, idx):
        return {
            'predicted_state': self.predicted_states[idx],
            # 'delta_pose': self.delta_poses[idx]
            'delta_pose': self.ground_truth_poses[idx]
        }
    
class PoseProber(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 6 for pose (x,y,z,roll,pitch,yaw)
        )
    
    def forward(self, x):
        return self.network(x)

def train_pose_prober(predicted_states, ground_truth_poses, device, 
                     num_epochs=100, batch_size=32, lr=1e-3):
    # Prepare dataset
    dataset = LocalizationProberData(predicted_states, ground_truth_poses, device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = predicted_states.shape[-1]
    prober = PoseProber(input_dim).to(device)
    optimizer = torch.optim.Adam(prober.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,1e-5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        prober.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            pred_states = batch['predicted_state']
            delta_poses = batch['delta_pose']
            
            pred_poses = prober(pred_states)
            pred_poses = pred_poses.unsqueeze(2)
            print(pred_poses.shape, delta_poses.shape)
            break
            loss = criterion(pred_poses, delta_poses)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        prober.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred_states = batch['predicted_state']
                delta_poses = batch['delta_pose']
                
                pred_poses = prober(pred_states)
                pred_poses = pred_poses.unsqueeze(2)
                loss = criterion(pred_poses, delta_poses)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(prober.state_dict(), 'best_pose_prober.pth')
        
        scheduler.step()
    
    return prober

if __name__ == "__main__":
    # Example usage with your JEPA model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your trained JEPA model
    jepa = models.JEPA(device=device)
    jepa.load_state_dict(torch.load('./jepa/1737687243.3784735/64_0.001_best_model.pth'))
    jepa.eval()

    data = data_structure.RajData("./data.csv", device)
    total_length = len(data)
    train_length = int(0.7 * total_length)
    val_length = int(0.15 * total_length)
    test_length = total_length - train_length - val_length
    train_dataset, val_dataset, test_dataset = random_split(data, [train_length, val_length, test_length])

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    
    # Get predicted states from JEPA
    with torch.no_grad():
        # Use your data loader to get a batch
        batch = next(iter(train_loader))
        imu = batch['imu'].to(device)
        delta_pose = batch['pose'].to(device)
        control_velocity = batch['velocity'].to(device)
        
        predicted_states = jepa(imu=imu, delta_pose=delta_pose, 
                              control_velocity=control_velocity)
    
    # Train the prober
    pose_prober = train_pose_prober(predicted_states, delta_pose, device)