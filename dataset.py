import torch.utils.data
import numpy as np
import nibabel as nib
import os

class SynDiffDataset(torch.utils.data.Dataset):
    def __init__(self, phase, input_path, contrast1='T1', contrast2='T2'):
        self.phase = phase
        self.input_path = input_path
        self.contrast1_files = [f for f in os.listdir(input_path) if f.endswith('.nii.gz') and contrast1 in f]
        self.contrast2_files = [f for f in os.listdir(input_path) if f.endswith('.nii.gz') and contrast2 in f]
        self.padding = True
        self.Norm = True
        # Ensure equal number of slices for simplicity (can sample randomly if unequal)
        self.num_slices = min(len(self.contrast1_files), len(self.contrast2_files)) * (100 // 2)  # Approx middle 50% of slices

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        file_idx1 = idx // (100 // 2)  # Assuming ~100 slices per volume, use middle 50
        slice_idx = idx % (100 // 2) + (100 // 4)  # Middle 50% slices
        file_idx2 = np.random.randint(0, len(self.contrast2_files))  # Random T2 for unpaired

        # Load T1 (target)
        file_path1 = os.path.join(self.input_path, self.contrast1_files[file_idx1])
        data1 = self.LoadDataSet(file_path1)
        x = data1[slice_idx % data1.shape[0]]  # Get specific slice

        # Load T2 (source/conditioning)
        file_path2 = os.path.join(self.input_path, self.contrast2_files[file_idx2])
        data2 = self.LoadDataSet(file_path2)
        y = data2[slice_idx % data2.shape[0]]  # Matching slice index

        return x, y

    def LoadDataSet(self, load_dir):
        img = nib.load(load_dir)
        data = img.get_fdata()  # 3D array [x, y, z]

        # Extract 2D slices
        slices = []
        for z in range(data.shape[2]):
            slice_data = data[:, :, z]
            slice_data = np.expand_dims(slice_data, axis=0)  # Add channel dim [1, H, W]
            slices.append(slice_data)
        data = np.stack(slices, axis=0)  # Shape [num_slices, 1, H, W]

        # Convert to float32
        data = data.astype(np.float32)

        if self.padding:
            pad_x = int((256 - data.shape[2]) / 2)
            pad_y = int((256 - data.shape[3]) / 2)
            print(f'padding in x-y with: {pad_x}-{pad_y}')
            data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

        if self.Norm:
            data = (data - np.mean(data)) / np.std(data)  # Zero mean, unit variance
            data = data * 2 - 1  # Scale to [-1, 1] to match test.py's range

        return data

# Update CreateDatasetSynthesis to use the new Dataset class
def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    return SynDiffDataset(phase, input_path, contrast1, contrast2)

# Example usage
if __name__ == "__main__":
    input_path = "path_to_your_ixi_data"
    dataset = CreateDatasetSynthesis("test", input_path, contrast1='T1', contrast2='T2')
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")