import torch
from torch import nn
import math

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
    
class DSConv(nn.Module):
    expansion = 1
    def __init__(self, f_3x3, f_1x1, kernel_size, stride=1, dilation=1, downsample=False, padding=1 , inplace = True):
        super(DSConv, self).__init__()
        self.relu = nn.ELU()
        
        
        self.depth_wise = nn.Conv1d(f_3x3, f_3x3,kernel_size=kernel_size,groups=f_3x3,stride=stride, padding = padding ,
                            bias=False)
                            
        
        self.bn_1 = nn.BatchNorm1d(f_3x3 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.point_wise = nn.Conv1d(f_3x3,f_1x1,kernel_size=1 ,bias=False)
        
        self.bn_2 = nn.BatchNorm1d(f_1x1 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       
        self.downsample_ = nn.Sequential( nn.Conv1d(f_3x3, f_1x1, kernel_size= 1, stride=stride, bias=False),
                                        nn.BatchNorm1d(f_1x1))
        self.downsample = downsample
        
        # self.elu_ = nn.ELU()
       
    def forward(self, x):
        residual = x
        # out = self.feature(x)
        out = self.depth_wise(x)
        out = self.bn_1 (out)
        out = self.relu (out)
        out = self.point_wise(out)
        out = self.bn_2 (out)
        out = self.relu(out)
        if self.downsample:
            residual = self.downsample_(x)

        out += residual.clone()
        out = self.relu(out)
        return out
    
class DSConv_Regular(nn.Module):
    expansion = 1
    def __init__(self, f_3x3, f_1x1, kernel_size, stride=1, dilation=1, downsample=False, padding=1 , inplace = True):
        super(DSConv_Regular, self).__init__()
        self.relu = nn.ELU()
        
        
        self.depth_wise = nn.Conv1d(f_3x3, f_3x3,kernel_size=kernel_size,groups=f_3x3,stride=stride, padding = padding ,
                            bias=False)
                            
        
        self.bn_1 = nn.BatchNorm1d(f_3x3 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.point_wise = nn.Conv1d(f_3x3,f_1x1,kernel_size=1 ,bias=False)
        
        self.bn_2 = nn.BatchNorm1d(f_1x1 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
       
    def forward(self, x):
        residual = x
        # out = self.feature(x)
        out = self.depth_wise(x)
        out = self.bn_1 (out)
        out = self.relu (out)
        out = self.point_wise(out)
        out = self.bn_2 (out)
        out = self.relu(out)
      

        out += residual.clone()
        out = self.relu(out)
        return out
class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(1, 1200))  # Learnable parameter with matching shape
        self.b = nn.Parameter(torch.zeros(1, 1200))  # Learnable parameter with matching shape

    def forward(self, TensorA, TensorB):
        return TensorA - self.W * TensorB + self.b
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)
    


    
class IMUNetOG(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
    
    
    def __init__(self, input_channels=6, output_size=2048, base_channels=64):
        # input_size: 1 x EEG channel x datapoint
        super(IMUNetOG, self).__init__()
        
        input_size = input_channels

        self.noise = CustomLayer()
        
        self.input_mult = 64
   
        
        self.input_block = nn.Sequential(
            nn.Conv1d(input_size, self.input_mult , kernel_size=7, stride=2, padding= 3, bias=False),
            nn.BatchNorm1d(self.input_mult, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1)
        )


        self.conv_1_1 = DSConv_Regular(self.input_mult, 64, 3 , downsample= False , padding=1,  stride = 1)
        self.conv_1_2 = DSConv_Regular( 64, 64, 3 , downsample= False , padding = 1)
       
        
        self.conv_3_1 = DSConv(64, 64, 3 , downsample= True , padding = 1 ,  stride = 1)
        self.conv_3_2 = DSConv_Regular( 64, 64, 3 , downsample= False , padding = 1 , stride = 1)
        
        
        self.conv_4_1 = DSConv(64, 128, 3 , downsample= True , padding = 1  , stride = 2)
        self.conv_4_2 = DSConv_Regular( 128, 128, 3 , downsample= False , padding = 1 , stride = 1)
        
        self.conv_5_1 = DSConv(128, 256, 3 , downsample= True , padding = 1,  stride = 2)
        self.conv_5_2 = DSConv_Regular( 256, 256, 3 , downsample= False , padding = 1 , stride = 1)
        
        
        self.conv_6_1 = DSConv(256, 512, 3 , downsample= True , padding = 1,  stride = 2)
        self.conv_6_2 = DSConv_Regular( 512, 512, 3 , downsample= False , padding = 1 , stride = 1)
        
        self.conv_7_1 = DSConv(512, 1024, 3 , downsample= True , padding = 1,  stride = 2)
        self.conv_7_2 = DSConv_Regular( 1024, 1024, 3 , downsample= False , padding = 1)
        
       
        
        self.relu = nn.GELU()
        
      
        
        self.output_block = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=400,
                      kernel_size=2, stride=1),
            nn.BatchNorm1d(400),
            
          )
        
        
        
        
        
        
        self.fc = nn.Sequential(
            nn.Linear(1200, 512)
            
            
        )
        
        self.projection_head = ProjectionHead(input_dim=512, hidden_dim=1024, output_dim=output_size)
        
        self.out_dim = output_size

    def forward(self, x):
      
        input_val = x.contiguous().view(x.size(0), -1)
        x = self.input_block(x)
        
        
       
        y = self.conv_1_1(x)

        y = self.conv_1_2(y)
        
        x = y
        y = self.conv_3_1(x)
       
        y = self.conv_3_2(y)
        
       
        y = self.conv_4_1(y)
        
        y = self.conv_4_2(y)
       
        y = self.conv_5_1(y)
        y = self.conv_5_2(y)
        
        y = self.conv_6_1(y)
        y = self.conv_6_2(y)
        
        y = self.conv_7_1(y)
        y = self.conv_7_2(y)
      
       
        out = self.output_block(y)
        out = out.view(out.size(0), -1)
        out = self.noise(out,input_val)
        out = self.relu(out)
        
        out_features = self.fc(out)
        
        out_projection = self.projection_head(out_features)
        return out_features, out_projection
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class IMUNetEncoder(nn.Module):
    def __init__(self, repr_dim=512):
        super().__init__()
        self.enc = IMUNetOG(input_channels=6, base_channels=64)
        self.enc_dim = self.enc.out_dim
        self.repr_dim = repr_dim
        self.fc = nn.Linear(self.enc_dim,self.repr_dim)

    def forward(self, data):
        _, out = self.enc(data)
        return self.fc(out)
    
class PoseEncoder(nn.Module):
    def __init__(self, repr_dim=512):
        super(PoseEncoder, self).__init__()
        
        input_dim = 6
        hidden_dims=[64, 256, 1024]
        self.input_dim = input_dim
        self.latent_dim = repr_dim
        
        self.L = 10
        self.fourier_dim = self.L * 2 * input_dim
        
        layers = []
        in_features = input_dim + self.fourier_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, repr_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def fourier_encode(self, p):
        encoded = []
        for l in range(self.L):
            encoded.extend([torch.sin(2**l * math.pi * p), torch.cos(2**l * math.pi * p)])
        return torch.cat(encoded, dim=-1)
    
    def forward(self, x):
        x = x.squeeze(dim=-1)
        check_nan_inf(x, "x_squeezed")

        # x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        x_normalized = torch.nn.functional.normalize(x, dim=-1)
        check_nan_inf(x_normalized, "x_normalized")

        x_fourier = self.fourier_encode(x_normalized)
        check_nan_inf(x_fourier, "x_fourier")

        x_combined = torch.cat([x_normalized, x_fourier], dim=-1)
        check_nan_inf(x_combined, "x_combined")

        latent = self.encoder(x_combined)
        check_nan_inf(latent, "latent")

        return latent

    
class ControlVelocityEncoder(nn.Module):
    def __init__(self, repr_dim=512):
        super().__init__()
        self.repr_dim = repr_dim
        self.enc = nn.Sequential(nn.Linear(2,self.repr_dim*2), nn.ReLU(),nn.Linear(self.repr_dim*2,self.repr_dim))

    def forward(self, data):
        return self.enc(data)
    
class CurrentStateEncoder(nn.Module):
    def __init__(self, repr_dim=512):
        super().__init__()
        self.repr_dim = repr_dim
        self.enc = nn.Sequential(
            nn.Linear(self.repr_dim*2, self.repr_dim*4),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(self.repr_dim*4, self.repr_dim)
        )

    def forward(self, data):
        return self.enc(data)

class StatePredictor(nn.Module):
    def __init__(self, repr_dim=512):
        super().__init__()
        self.repr_dim = repr_dim
        self.predictor = nn.LSTMCell(self.repr_dim*2, self.repr_dim*4)
        self.fc = nn.Sequential(nn.Linear(self.repr_dim*4, self.repr_dim), nn.BatchNorm1d(self.repr_dim))

    def forward(self, data):
        pred, _ = self.predictor(data)
        out = self.fc(pred)
        return out
    
class ErrorCorrector(nn.Module):
    def __init__(self, repr_dim=512):
        super().__init__()
        self.repr_dim = repr_dim
        self.enc = nn.Sequential(nn.Linear(self.repr_dim*2, self.repr_dim*4), nn.ReLU(),nn.Linear(self.repr_dim*4, self.repr_dim))

    def forward(self, data):
        return self.enc(data)
    
class JEPA(nn.Module):
    def __init__(self, device='cuda', repr_dim=512):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.imu_encoder = IMUNetEncoder(self.repr_dim)
        self.pose_encoder = PoseEncoder(self.repr_dim)
        self.control_velocity_encoder = ControlVelocityEncoder(self.repr_dim)
        self.current_state_encoder = CurrentStateEncoder(self.repr_dim)
        self.state_predictor = StatePredictor(self.repr_dim)
        self.error_corrector = ErrorCorrector(self.repr_dim)
        self.set_device()

    def forward(self, imu, delta_pose, control_velocity):
        B, T, S, C = imu.shape
        t0_imu_encoding = self.imu_encoder(imu[:, 0].reshape(-1, C, S))
        check_nan_inf(t0_imu_encoding, "t0_imu_encoding")
        check_nan_inf(delta_pose[:, 0].reshape(-1, 6, 1), "delta_pose")
        t0_delta_pose_encoding = self.pose_encoder(delta_pose[:, 0].reshape(-1, 6, 1))
        check_nan_inf(t0_delta_pose_encoding, "t0_delta_pose_encoding")
        control_velocity_encodings = self.control_velocity_encoder(control_velocity.reshape(-1, 2)).reshape(B, T-1, -1)
        check_nan_inf(control_velocity_encodings, "control_velocity_encodings")
        initial_state = self.current_state_encoder(torch.cat((t0_imu_encoding, t0_delta_pose_encoding), dim=1))
        check_nan_inf(initial_state, "initial_state")
        predicted_states = [initial_state.unsqueeze(1)]
        next_pred_state = self.state_predictor(torch.cat((initial_state, control_velocity_encodings[:, 0]), dim=1))
        check_nan_inf(next_pred_state, "next_pred_state")
        past_pred_state = next_pred_state

        for t in range(1, T-1):
            current_imu_encoding = self.imu_encoder(imu[:, t].reshape(-1, C, S))
            error_corrected_current_state = self.error_corrector(torch.cat((past_pred_state, current_imu_encoding), dim=1))
            predicted_states.append(error_corrected_current_state.unsqueeze(1))
            next_pred_state = self.state_predictor(torch.cat((error_corrected_current_state, control_velocity_encodings[:, t]), dim=1))
            past_pred_state = next_pred_state
        
        last_timestep_imu_encoding = self.imu_encoder(imu[:, -1].reshape(-1, C, S))
        error_corrected_last_timestep_state = self.error_corrector(torch.cat((past_pred_state, last_timestep_imu_encoding), dim=1))
        predicted_states.append(error_corrected_last_timestep_state.unsqueeze(1))

        predictions = torch.cat(predicted_states, dim=1)
        return predictions
    
    def set_device(self):
        self.to(self.device)
        self.imu_encoder.to(self.device)
        self.pose_encoder.to(self.device)
        self.control_velocity_encoder.to(self.device)
        self.current_state_encoder.to(self.device)
        self.state_predictor.to(self.device)
        self.error_corrector.to(self.device)