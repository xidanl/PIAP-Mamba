import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from RM_physics_modules import PhysicsCalc

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CBR, self).__init__()
        
        padding = (dilation * (kernel_size - 1)) // 2  
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
            
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Mamba_model(nn.Module):
    """
    The model architecture for rebuilding iAPs from eAPs.

    Parameters:
    - seq_len: recording time step number for action potentials
    - n_layers: the number of mamba-2 layers stacked in the model
    - dim: model dimention
    - physics / use_cond: controlling physics constraint
    """
    def __init__(self, seq_len=8000, n_layers=4,dim=512,physics=True,use_cond=False):
        super().__init__()

        self.physics = physics
        self.use_cond = use_cond

        self.in_conv1 = CBR(1, dim // 4, kernel_size=1)
        self.in_conv2 = CBR(1, dim // 4, kernel_size=5)
        self.in_conv3 = CBR(1, dim // 4, kernel_size=9)
        self.in_conv4 = CBR(1, dim // 4, kernel_size=13)
        
        self.input_norm = nn.LayerNorm(dim)

        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=dim, d_state=32, d_conv=4, expand=2, layer_idx=i, rmsnorm=True) 
            for i in range(n_layers)
        ])


        self.out_proj = nn.Linear(dim, 1)
        self.tanh = nn.Tanh()

        if self.physics:
            
            self.v_out_enc = nn.Sequential(
                CBR(in_channels=1, out_channels=32, kernel_size=11, stride=5, dilation=1),
                CBR(in_channels=32, out_channels=64, kernel_size=11, stride=5, dilation=1),
                CBR(in_channels=64, out_channels=96, kernel_size=11, stride=5, dilation=1),
                CBR(in_channels=96, out_channels=128, kernel_size=11, stride=5, dilation=1),
                nn.AdaptiveAvgPool1d(1),
            )
            
            self.b_enc = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 8),
                nn.ReLU(inplace=True),
            )
            
            self.physics_calc_module = PhysicsCalc(input_dim=8)  
    
    def forward(self,x):

        x = x.unsqueeze(1)
        x = torch.cat([self.in_conv1(x), self.in_conv2(x), self.in_conv3(x), self.in_conv4(x)],dim=1)

        x = x.transpose(1,2)
        y = self.input_norm(x)

        for mamba_layer in self.mamba_layers:
            y = mamba_layer(y)  

        out = self.tanh(self.out_proj(y))
        out = out.squeeze(-1)

        dv_out = None
        if self.physics:
            v_out = out.unsqueeze(1)
            bk = self.v_out_enc(v_out).squeeze(-1)
            b = self.b_enc(bk)
            
            dv_out = self.physics_calc_module(v_out,b,self.use_cond,'noq')[0]
            dv_out = dv_out.squeeze(1)

        return out, dv_out

        

