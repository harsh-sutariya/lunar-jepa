import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import train_localization_prober
import models
import matplotlib.pyplot as plt


def test_pose_prober(prober, jepa, test_loader, device):
    prober.eval()
    jepa.eval()

    criterion = torch.nn.MSELoss()
    test_loss = 0
    
    all_actuals = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Prober"):
            # Get batch data
            imu = batch['imu'].to(device)
            delta_pose = batch['pose'].to(device)
            control_velocity = batch['velocity'].to(device)
            # next_pose = batch['next_pose'].to(device)
            
            # Get JEPA embeddings
            predicted_states = jepa(imu=imu, delta_pose=delta_pose, 
                                 control_velocity=control_velocity)
            
            # Get prober predictions
            pred_poses = prober(predicted_states)
            pred_poses = pred_poses.unsqueeze(2)
            
            # Calculate loss
            loss = criterion(pred_poses, delta_pose)
            test_loss += loss.item()
            
            # Store predictions and actuals
            all_actuals.append(delta_pose.cpu())
            all_predictions.append(pred_poses[:, :-1, :].cpu())
    
    test_loss /= len(test_loader)
    print(f"Average Test Loss: {test_loss:.6f}")
    
    return all_actuals, all_predictions

def visualize_predictions(all_actuals, all_predictions):
    # Combine all batches and reshape
    actuals = torch.cat(all_actuals, dim=0).squeeze()
    predictions = torch.cat(all_predictions, dim=0).squeeze()
    
    # Trim actuals to match predictions
    actuals = actuals[:, :4, :]
    
    # Print final shapes after concatenation and trimming
    print(f"Combined actuals shape: {actuals.shape}")
    print(f"Combined predictions shape: {predictions.shape}")
    
    # Get the actual number of dimensions
    num_dims = actuals.shape[2]
    pose_dimensions = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'][:num_dims]
    
    # Create a figure with appropriate number of subplots
    rows = (num_dims + 1) // 2  # Calculate needed rows
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
    fig.suptitle("Actual vs Predicted Poses for Each Dimension", fontsize=16)
    
    # Make axes iterable even if there's only one row
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ax in enumerate(axes.flat):
        if i < num_dims:
            actual_values = actuals[:, :, i].numpy().flatten()
            predicted_values = predictions[:, :, i].numpy().flatten()
            
            ax.plot(actual_values, label='Actual', color='blue', alpha=0.7)
            ax.plot(predicted_values, label='Predicted', color='red', alpha=0.7)
            
            ax.set_title(f"{pose_dimensions[i].capitalize()} Dimension")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value")
            ax.legend()
        else:
            fig.delaxes(ax)  # Remove unused subplots
    
    plt.tight_layout()
    plt.show()


def print_x_coordinates(all_actuals, all_predictions):
    # Combine all batches
    actuals = torch.cat(all_actuals, dim=0).squeeze()
    predictions = torch.cat(all_predictions, dim=0).squeeze()
    
    # Extract first two x coordinates
    actual_x = actuals[:2, 1, 0].numpy()
    predicted_x = predictions[:2, 1, 0].numpy()
    
    print(f"First two actual x coordinates: {actual_x}")
    print(f"First two predicted x coordinates: {predicted_x}")






if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    jepa = models.JEPA(device=device)
    jepa.load_state_dict(torch.load('./jepa/1737687243.3784735/64_0.001_best_model.pth'))


    input_dim = jepa.repr_dim
    prober = train_localization_prober.PoseProber(input_dim).to(device)
    prober.load_state_dict(torch.load('../weights/pose_prober_gpu.pth'))
    
    # Prepare test data
    data = train_localization_prober.RajData("./data.csv", device)
    total_length = len(data)
    train_length = int(0.7 * total_length)
    val_length = int(0.15 * total_length)
    test_length = total_length - train_length - val_length
    _, _, test_dataset = random_split(data, [train_length, val_length, test_length])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False)
    
    # Run testing
    all_actuals, all_predictions = test_pose_prober(prober, jepa, test_loader, device)
    
    visualize_predictions(all_actuals, all_predictions)
    print_x_coordinates(all_actuals, all_predictions)