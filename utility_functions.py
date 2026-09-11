import torch
import matplotlib.pyplot as plt

def denormalize_image(tensor_image):
    tensor_image = (tensor_image + 1.0) / 2.0
    return torch.clamp(tensor_image, 0.0, 1.0)

def show_image(image):
    if len(image.shape)==4:
        image=image.squeeze(0)
    if image.min() < 0.0 or image.max() > 1.0:
        image = denormalize_image(image)
    image=image.permute(1,2,0).cpu().numpy()
    plt.figure(figsize=(3,3))
    plt.imshow(image,interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
def reconstruct_image(model,image):
    model.eval()
    if len(image.shape)==3:
        image=image.unsqueeze(0)
    image=image.to(next(model.parameters()).device)
    with torch.no_grad():
        x_rec, _ , _=model(image)
    show_image(image)
    show_image(x_rec)

def interpolate_images(model, img1, img2, steps=10, device='cpu'):
    model.eval()
    model.to(device)
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        alphas = torch.linspace(0, 1, steps=steps).to(device)
        z_interp = torch.zeros((steps, mu1.shape[1])).to(device)
        for i, alpha in enumerate(alphas):
            z_interp[i] = (1 - alpha) * mu1 + alpha * mu2
        generated_images = model.decode(z_interp).cpu()
    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i in range(steps):
        img = generated_images[i]
        if img.min() < 0.0 or img.max() > 1.0:
            img = denormalize_image(img)
        img_plot = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img_plot, interpolation='bilinear')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title("Inizio")
        elif i == steps - 1:
            axes[i].set_title("Fine")
    plt.tight_layout()
    plt.show()

def perturb_latent(model,z_vector,latent_idx,value):
  z=z_vector.clone()
  z[0,latent_idx]+=value
  with torch.no_grad():
        img_rec = model.decode(z)
  show_image(img_rec.squeeze(0).cpu())

def encode_image(model,image,device='cpu'):
    model.eval()
    model.to(device)
    image=image.unsqueeze(0).to(device)
    with torch.no_grad():
        mu,_=model.encode(image)
    return mu

def compute_latent_statistics(model, dataloader, device, num_batches=10):
    model.eval()
    model.to(device)
    all_mus = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            mu, _ = model.encode(batch.to(device))
            all_mus.append(mu)
    all_mus = torch.cat(all_mus, dim=0)
    mu_mean = torch.mean(all_mus, dim=0)
    std_mean = torch.std(all_mus, dim=0)
    return mu_mean.cpu(), std_mean.cpu()

def test_all_latents(model, z_vector,mu_mean=None,std_mean=None, start=-3, end=3, n_steps=10,device='cpu'):
    model.eval()
    model.to(device)
    steps=torch.linspace(start,end,n_steps)
    n_righe = model.latent_dim
    n_colonne = len(steps)
    fig, axes = plt.subplots(n_righe, n_colonne, figsize=(n_colonne * 2, n_righe * 2.2))
    if n_righe == 1: axes = [axes]
    if n_colonne == 1: axes = [[ax] for ax in axes]
    with torch.no_grad():
        for i in range(n_righe):
            for j, val in enumerate(steps):
                z = z_vector.clone().to(device)
                if mu_mean is None:
                    z[0, i] +=val
                else:     
                    z[0, i] += mu_mean[i] + (stds[i]*val)
                img_rec = model.decode(z)
                if isinstance(img_rec, tuple):
                    img_rec = img_rec[0]
                img_rec = denormalize_image(img_rec)
                img_vis = img_rec.squeeze(0).permute(1, 2, 0).cpu().numpy()
                ax = axes[i][j]
                ax.imshow(img_vis)
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"Val: {val}")
                if j == 0:
                    ax.text(-0.2, 0.5, f"Idx {i}", transform=ax.transAxes, va='center', ha='right', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()