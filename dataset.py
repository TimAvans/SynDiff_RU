import torch.utils.data
import numpy as np
import nibabel as nib
import os
import tarfile
import tempfile
from tqdm import tqdm

class SynDiffDataset(torch.utils.data.Dataset):
    def __init__(self, phase, input_path, contrast1='T1', contrast2='T2'):
        self.phase = phase
        self.input_path = input_path
        # Open the tar files for T1 and T2
        self.tar_t1 = tarfile.open(os.path.join(input_path, f'IXI-{contrast1}.tar'), 'r')
        self.tar_t2 = tarfile.open(os.path.join(input_path, f'IXI-{contrast2}.tar'), 'r')
        self.contrast1_files = [f for f in self.tar_t1.getnames() if f.endswith('.nii.gz') and contrast1 in os.path.basename(f)]
        self.contrast2_files = [f for f in self.tar_t2.getnames() if f.endswith('.nii.gz') and contrast2 in os.path.basename(f)]
        self.padding = True
        self.Norm = True
        # Dynamically determine slice count from the first valid file
        self.num_slices = 0
        if self.contrast1_files:
            try:
                first_file = self.tar_t1.extractfile(self.contrast1_files[0])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
                    tmp_file.write(first_file.read())
                    tmp_file_path = tmp_file.name
                img = nib.load(tmp_file_path)
                os.unlink(tmp_file_path)
                num_slices = img.shape[2]
                self.num_slices = min(len(self.contrast1_files), len(self.contrast2_files)) * (num_slices // 2)
            except Exception as e:
                print(f"Error determining slice count: {e}")

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        if not self.contrast1_files or not self.contrast2_files:
            raise ValueError("No valid files found in tar archives.")
        file_idx1 = idx // (100 // 2)  # Assuming ~100 slices per volume, use middle 50
        slice_idx = idx % (100 // 2) + (100 // 4)  # Middle 50% slices
        file_idx2 = np.random.randint(0, len(self.contrast2_files))  # Random T2 for unpaired

        # Load T1 (target) from tar
        file_path1 = self.contrast1_files[file_idx1 % len(self.contrast1_files)]
        file1 = self.tar_t1.extractfile(file_path1)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            try:
                tmp_file.write(file1.read())
                tmp_file_path = tmp_file.name
                data1 = self.LoadDataSet(tmp_file_path)
                x = data1[slice_idx % data1.shape[0]]  # Get specific slice
            except Exception as e:
                print(f"Error loading T1 file {file_path1}: {e}")
                return self.__getitem__((idx + 1) % self.num_slices)  # Retry with next index
            finally:
                os.unlink(tmp_file_path)

        # Load T2 (source/conditioning) from tar
        file_path2 = self.contrast2_files[file_idx2]
        file2 = self.tar_t2.extractfile(file_path2)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            try:
                tmp_file.write(file2.read())
                tmp_file_path = tmp_file.name
                data2 = self.LoadDataSet(tmp_file_path)
                y = data2[slice_idx % data2.shape[0]]  # Matching slice index
            except Exception as e:
                print(f"Error loading T2 file {file_path2}: {e}")
                return self.__getitem__((idx + 1) % self.num_slices)  # Retry with next index
            finally:
                os.unlink(tmp_file_path)

        return x, y

    def LoadDataSet(self, file_path):
        # Use tqdm to show progress for slice extraction
        with tqdm(total=data.shape[2], desc="Extracting slices") as pbar:
            img = nib.load(file_path)
            data = img.get_fdata()  # 3D array [x, y, z]

            # Extract 2D slices
            slices = []
            for z in range(data.shape[2]):
                slice_data = data[:, :, z]
                slice_data = np.expand_dims(slice_data, axis=0)  # Add channel dim [1, 256, 256]
                slices.append(slice_data)
                pbar.update(1)
            data = np.stack(slices, axis=0)  # Shape [num_slices, 1, 256, 256]

            # Convert to float32
            data = data.astype(np.float32)

            if self.padding:
                pad_x = int((256 - data.shape[2]) / 2)
                pad_y = int((256 - data.shape[3]) / 2)
                if pad_x > 0 or pad_y > 0:
                    pbar.set_description("Padding slices")
                    data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

            if self.Norm:
                pbar.set_description("Normalizing data")
                data = (data - np.mean(data)) / np.std(data)  # Zero mean, unit variance
                data = data * 2 - 1  # Scale to [-1, 1] to match test.py's range

        return data

    def __del__(self):
        if hasattr(self, 'tar_t1'):
            self.tar_t1.close()
        if hasattr(self, 'tar_t2'):
            self.tar_t2.close()

# Update CreateDatasetSynthesis to use the new Dataset class
def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    return SynDiffDataset(phase, input_path, contrast1, contrast2)

# Example usage
if __name__ == "__main__":
    input_path = "/content/drive/My Drive/IXI"
    dataset = CreateDatasetSynthesis("test", input_path, contrast1='T1', contrast2='T2')
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")