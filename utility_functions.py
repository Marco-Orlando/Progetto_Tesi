import torch
import matplotlib.pyplot as plt

def denormalize_image(tensor_image):
    tensor_image = (tensor_image + 1.0) / 2.0
    return torch.clamp(tensor_image, 0.0, 1.0)

def show_image(image):
  if image.min() < 0.0 or image.max() > 1.0:
        image = denormalize_image(image)
  image=image.permute(1,2,0).numpy()
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
  show_image(image.squeeze(0).cpu())
  show_image(x_rec.squeeze(0).cpu())