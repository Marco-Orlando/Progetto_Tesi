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