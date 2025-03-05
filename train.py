from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import wandb
from tqdm import tqdm
import time
import os
from models import JEPA
from data_structure import RajData

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"NaN indices: {torch.nonzero(torch.isnan(tensor))}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        print(f"Inf indices: {torch.nonzero(torch.isinf(tensor))}")
        return True
    return False


def compute_bt_loss(embed1, embed2, device, lam=0.005, eps = 1e-8):
    """
    e1: (B, T-1, D)
    e2: (B, T-1, D)
    """
    B, T, D = embed1.shape
    if check_nan_inf(embed1, "embed1") or check_nan_inf(embed2, "embed2"):
        print("returning due to nan values")
    normalized_embed1 = (embed1 - embed1.mean(0))/(embed1.std(0)+eps) # B, T-1, D
    normalized_embed2 = (embed2 - embed2.mean(0))/(embed2.std(0)+eps) # B, T-1, D
    
    corr_matrix = torch.bmm(normalized_embed1.permute(1,2,0),normalized_embed2.permute(1,0,2))/B # T-1, D, D
    c_diff = (corr_matrix - torch.eye(D).reshape(1,D, D).repeat(T,1,1).to(device)).pow(2)
    off_diagonal = (torch.ones((D, D))-torch.eye(D)).reshape(1,D,D).repeat(T,1,1).to(device)
    c_diff *= (off_diagonal*lam + torch.eye(D).reshape(1,D, D).repeat(T,1,1).to(device))
    return c_diff.sum()

def train(jepa, train_dataloader, optimizer, device):
    jepa.train()
    total_loss = 0
    for idx, batch in tqdm(enumerate(train_dataloader)):
        imu = batch['imu'].to(device)
        delta_pose = batch['pose'].to(device)
        control_velocity = batch['velocity'].to(device)
        pred_states = jepa(imu=imu, delta_pose=delta_pose, control_velocity=control_velocity)
        B,T,S,C = imu.shape
        actual_imu_encoding = jepa.imu_encoder(imu.reshape(-1, C, S))
        actual_delta_pose_encoding = jepa.pose_encoder(delta_pose.reshape(-1, 6, 1))
        actual_states = jepa.current_state_encoder(torch.cat((actual_imu_encoding, actual_delta_pose_encoding), dim=1)).reshape(B,T,-1)
        loss = compute_bt_loss(pred_states[:,1:],actual_states[:,1:],device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"train_batch_loss":loss})
    return total_loss/(idx+1)

def val(jepa, val_dataloader, device):
    jepa.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader)):
            imu = batch['imu'].to(device)
            delta_pose = batch['pose'].to(device)
            control_velocity = batch['velocity'].to(device)
            pred_states = jepa(imu=imu, delta_pose=delta_pose, control_velocity=control_velocity)
            B,T,S,C = imu.shape
            actual_imu_encoding = jepa.imu_encoder(imu.reshape(-1, C, S))
            actual_delta_pose_encoding = jepa.pose_encoder(delta_pose.reshape(-1, 6, 1))
            actual_states = jepa.current_state_encoder(torch.cat((actual_imu_encoding, actual_delta_pose_encoding), dim=1)).reshape(B,T,-1)
            loss = compute_bt_loss(pred_states[:,1:],actual_states[:,1:],device)
            total_loss += loss.item()
            wandb.log({"val_batch_loss":loss})
        return total_loss/(idx+1)
    
def main():

    wandb.init(project="lunar_jepa")
    torch.autograd.set_detect_anomaly(True)

    device=get_device()

    start_time = time.time()
    save_dir = os.path.join("./jepa",str(start_time))
    print(f'Saving Dir: {save_dir}')
    os.makedirs(save_dir)

    data = RajData("./data.csv", device)
    total_length = len(data)
    train_length = int(0.7 * total_length)
    val_length = int(0.15 * total_length)
    test_length = total_length - train_length - val_length
    train_dataset, val_dataset, test_dataset = random_split(data, [train_length, val_length, test_length])

    print("Train Length: ", len(train_dataset), "Val Length: ", len(val_dataset))

    batch_size = 64
    lr=1e-3
    weight_decay=1e-5
    num_epochs = 100

    wandb.config.update({
    "device": str(device),
    "batch_size": batch_size,
    "lr": lr,
    "weight_decay": weight_decay,
    "num_epochs": num_epochs
    })

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    jepa = JEPA(device=device)
    
    optimizer = optim.Adam(jepa.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,1e-5)
    torch.nn.utils.clip_grad_norm_(jepa.parameters(), max_norm=1.0)

    best_val_loss = float('inf')

    wandb.watch(jepa, log="all")
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss",step_metric="epoch")
    wandb.define_metric("val_loss",step_metric="epoch")

    for epoch in tqdm(range(num_epochs)):
        train_loss = train(jepa,train_loader,optimizer, device)
        print(f'Train Loss: {train_loss} Epoch: {epoch}')
        val_loss = val(jepa,val_loader,device)
        print(f'Val Loss: {val_loss} Epoch: {epoch}')
        wandb.log({"val_loss":val_loss,"train_loss":train_loss, "epoch":epoch})
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(jepa.state_dict(),
                    f'{save_dir}/{batch_size}_{lr}_best_model.pth')
        torch.save(jepa.state_dict(),
                    f'{save_dir}/{batch_size}_{lr}_epoch_{epoch}.pth')
        scheduler.step()
    
    print(f'Total Time: {time.time()-start_time}')
    wandb.finish()

if __name__ == "__main__":
    main()