import os
import h5py
import torch
from torch.utils.data import IterableDataset
import kagglehub
import numpy as np

class Shapes3DDataset(IterableDataset):
    def __init__(self, chunk_size=10000, transform=None):
        root_path = kagglehub.dataset_download('kleinli2019gmailcom/shapes3d')
        h5_files = os.path.join(root_path,"3dshapes.h5")
        if not os.path.exists(h5_files):
            raise FileNotFoundError("File .h5 non trovato.")
        self.file_path = h5_files
        self.chunk_size = chunk_size
        self.transform = transform

        with h5py.File(self.file_path, 'r') as f:
            self.total_samples = f['images'].shape[0]

        print(f"Dataset inizializzato: {self.total_samples} immagini totali. Chunk size: {chunk_size}")
        
    def __len__(self):
      return self.total_samples
        
    def __iter__(self):
        with h5py.File(self.file_path, 'r') as f:
            dataset_images = f['images']

            for start_idx in range(0, self.total_samples, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, self.total_samples)

                chunk = dataset_images[start_idx:end_idx]

                chunk_length = len(chunk)
                local_indices = torch.randperm(chunk_length).tolist()

                for idx in local_indices:
                    img = chunk[idx]

                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                    if self.transform:
                        img = self.transform(img)

                    yield img
