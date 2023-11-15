from easydict import EasyDict
import torch
import torch.nn as nn
from models.backbones.dfformer import DFformer


class EEGAutoEncoder(nn.Module):
    def __init__(self, args: EasyDict):
        super().__init__()
        
        self.encoder = DFformer(args)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, args.dim, kernel_size=(1, 4), stride=(1, 2)),
            nn.GELU(),
            nn.Conv2d(args.dim, args.dim, kernel_size=(1, 4), padding='same'),
            nn.GELU(),
            nn.ConvTranspose2d(args.dim, args.dim, kernel_size=(1, 8), stride=(1, 4)),
            nn.GELU(),
            nn.Conv2d(args.dim, args.dim, kernel_size=(1, 8), padding='same'),
            nn.GELU(),
            nn.ConvTranspose2d(args.dim, args.dim, kernel_size=(1, 125)),
            nn.GELU(),
            nn.Conv2d(args.dim, 1, kernel_size=(1, 125), padding='same'),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        
        x = self.encoder(x)
        temp_token = x[:, 0, :, 1:]
        spatial_token = x[:, 1:, :, 0]
        f = torch.bmm(spatial_token, temp_token)
        f = f.unsqueeze(1)
        
        f = self.decoder(f)
        f = f.transpose(1, 2)
        
        return f
    