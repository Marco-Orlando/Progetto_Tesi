from torch import nn
import torch 

class VAE(nn.Module):
  def __init__(self,latent_dim=6):
    super().__init__()
    self.latent_dim=latent_dim
    self.encoder=nn.Sequential(
        # (3,64,64) -> (32,64,64)
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        # (32,64,64) -> (32,32,32)
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        # (32,32,32) -> (64,16,16)
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        # (64,16,16) -> (128,8,8)
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        # (128,8,8) -> (256,4,4)
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Flatten()
    )
    self.hid_dim=256*4*4
    self.mu_layer=nn.Linear(self.hid_dim,latent_dim)
    self.logvar_layer=nn.Linear(self.hid_dim,latent_dim)
    self.decoder=nn.Sequential(
        nn.Linear(latent_dim,self.hid_dim),
        nn.Unflatten(dim=1,unflattened_size=(256,4,4)),
        # (256,4,4) -> (256,8,8)
        nn.Upsample(size=(8,8),mode='bilinear',align_corners=False),
        # (256,8,8) -> (128,8,8)
        nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        # (128,8,8) -> (128,16,16)
        nn.Upsample(size=(16,16),mode='bilinear',align_corners=False),
        # (128,16,16) -> (64,16,16)
        nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        # (64,16,16) -> (64,32,32)
        nn.Upsample(size=(32,32),mode='bilinear',align_corners=False),
        # (64,32,32) -> (32,32,32)
        nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        # (32,32,32) -> (32,64,64)
        nn.Upsample(size=(64,64),mode='bilinear',align_corners=False),
        # (32,64,64) -> (3,64,64)
        nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,stride=1,padding=1),
        nn.Tanh()
    )
   
  def reparamiter(self,mu,logvar):
    std=torch.exp(0.5*logvar)
    eps=torch.randn_like(std)
    return mu+std*eps

  def encode(self,x):
    h=self.encoder(x)
    mu=self.mu_layer(h)
    logvar=self.logvar_layer(h)
    return mu,logvar

  def decode(self,z):
    return self.decoder(z)

  def forward(self,x):
      mu,logvar=self.encode(x)
      z=self.reparamiter(mu,logvar)
      x_rec=self.decode(z)
      return x_rec, mu, logvar