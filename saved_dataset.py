import os
import numpy as np
from torch.utils.data import Dataset
import torch

class SavedDataset(Dataset):
    def __init__(self, file_path):
        """
        Initialize the dataset with a single .npy file containing preprocessed (x, y) pairs.
        
        Args:
            file_path (str): Path to the .npy file (e.g., /content/drive/MyDrive/IXI_processed_dataset_full.npy).
        """
        self.file_path = file_path
        # Load the data once and store it (assuming it fits in memory; use lazy loading if needed)
        self.data = np.load(file_path, allow_pickle=True)
        self.num_samples = len(self.data)
        print(f"Loaded {self.num_samples} samples from {file_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return the (x, y) pair at the given index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (x, y) as torch tensors.
        """
        x, y = self.data[idx]
        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y

# Example usage
if __name__ == "__main__":
    dataset = SavedDataset("/content/drive/MyDrive/IXI_processed_dataset_full.npy")
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample shapes: x={x.shape}, y={y.shape}")