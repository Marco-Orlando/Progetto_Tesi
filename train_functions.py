import torch.nn.functional as F
import torch
from tqdm import tqdm
import os 
from utility_functions import reconstruct_image


                                                ##--------VANILLA_VAE-------##


def loss_function(x_rec,x_real,mu,logvar,beta=1.0):
  REC_LOSS=torch.mean(torch.sum(F.mse_loss(x_rec,x_real,reduction='none'),dim=[1, 2, 3]))
  KL_DIV=-0.5*torch.mean(torch.sum(1-logvar.exp()-mu.pow(2)+logvar,dim=1))
  TOT_LOSS=REC_LOSS+beta*KL_DIV
  return TOT_LOSS,REC_LOSS,KL_DIV

def train_vae(model,data_loader,optimizer,criterion,n_epochs,device,test_image=None,checkpoint_file='checkpoint.pth',print_results=False,time_print=2):
  model.train()
  model.to(device)
  print(f'Addestramento avviato su: {device}')

  start_epoch=0

  if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch=checkpoint['epoch']+1

  for epoch in range(start_epoch,n_epochs):
    progress=tqdm(data_loader,desc=f'Epoch:{epoch+1}/{n_epochs}')
    losses=0.0
    for x_batch in progress:
      x_batch=x_batch.to(device)
      x_rec,mu,logvar=model(x_batch)
      tot_loss,rc_loss,kl_div=criterion(x_rec,x_batch,mu,logvar)
      optimizer.zero_grad()
      tot_loss.backward()
      #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      progress.set_postfix({'Tot_loss': tot_loss.item(), 'Rec_loss': rc_loss.item(), 'KL_div': kl_div.item()})
      losses+=tot_loss.item()
    epoch_loss=losses/len(data_loader)
    checkpoint={
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': tot_loss.item()
    }
    print(f'||-- Loss mean: {epoch_loss}')
    if print_results:
        if (epoch+1)%time_print==0:
          reconstruct_image(model,test_image)
    torch.save(checkpoint,f'{checkpoint_file}')
    print(f'Modello salvato corretamente ( Epoca: {epoch} )')

##------------BETA_VAE---------------##
import math

def get_current_beta(current_step,max_beta=10,min_beta=0,steps=10,t_0=5,k=0.005,waiting=1,max_time=1,mode='linear'):
    if current_step<waiting:
        current_beta=min_beta
    elif current_step>max_time:
        current_beta=max_beta
    else:
        if mode=='linear':
            betas=torch.linspace(min_beta,max_beta,(steps-(waiting+max_time)))
            current_beta=betas[current_step]
            current_beta=current_beta.item()
        elif mode=='sigmoid':
            sigmid=1/(1+math.exp(-k*(current_step-t_0)))
            current_beta = max_beta + (max_beta - min_beta) * sigmoid
        else:
            current_beta=1.0
    
    return current_beta

def train_vae_beta(model,data_loader,optimizer,criterion,n_epochs,device,test_image=None,checkpoint_file='checkpoint_beta.pth',print_results=False,time_print=2):
  model.train()
  model.to(device)
  print(f'Addestramento avviato su: {device}')

  start_epoch=0

  if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch=checkpoint['epoch']+1

  for epoch in range(start_epoch,n_epochs):
    progress=tqdm(data_loader,desc=f'Epoch:{epoch+1}/{n_epochs}')
    losses=0.0
    current_beta=get_current_beta(epoch,steps=n_epochs,max_beta=10,min_beta=0,t_0=(n_epochs/2),k=0.005,mode='linear')
    for x_batch in progress:
      x_batch=x_batch.to(device)
      x_rec,mu,logvar=model(x_batch)
      tot_loss,rc_loss,kl_div=criterion(x_rec,x_batch,mu,logvar,current_beta)
      optimizer.zero_grad()
      tot_loss.backward()
      optimizer.step()
      progress.set_postfix({'Tot_loss': tot_loss.item(), 'Rec_loss': rc_loss.item(), 'KL_div': kl_div.item(),'Beta': current_beta})
      losses+=tot_loss.item()
    epoch_loss=losses/len(data_loader)
    checkpoint={
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': tot_loss.item()
    }
    print(f'||-- Loss mean: {epoch_loss}')
    if print_results:
        if (epoch+1)%time_print==0:
          reconstruct_image(model,test_image)
    torch.save(checkpoint,f'{checkpoint_file}')
    print(f'Modello salvato corretamente ( Epoca: {epoch} )')