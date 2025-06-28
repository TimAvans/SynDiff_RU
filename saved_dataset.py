import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CreateDatasetSynthesis(Dataset):
    def __init__(self, phase, root_dir, contrast1="T1", contrast2="PD", max_slices=None):
        """
        Args:
            phase (str): 'train', 'test', etc. (not used here but can be used to split sets)
            root_dir (str): Directory with *_T1_filtered.npy and *_aligned_PD_filtered.npy files
            contrast1 (str): Input contrast (e.g., "T1")
            contrast2 (str): Target contrast (e.g., "PD")
            max_slices (int or None): Max number of slices to load
        """
        self.samples = []
        self.contrast1 = contrast1
        self.contrast2 = contrast2

        all_files = sorted(os.listdir(root_dir))
        subjects = sorted(set(f.split("_")[0] for f in all_files if f.endswith(f"{contrast1}_filtered.npy")))

        for subj_id in subjects:
            file1 = os.path.join(root_dir, f"{subj_id}_{contrast1}_filtered.npy")
            file2 = os.path.join(root_dir, f"{subj_id}_aligned_{contrast2}_filtered.npy")

            if not os.path.exists(file1) or not os.path.exists(file2):
                continue

            vol1 = np.load(file1)
            vol2 = np.load(file2)

            # Make sure number of slices matches
            total = min(len(vol1), len(vol2))
            for z in range(total):
                self.samples.append((vol1[z], vol2[z]))

        if max_slices is not None and len(self.samples) > max_slices:
            self.samples = self.samples[:max_slices]

        print(f"[{phase}] Loaded {len(self.samples)} slices with contrasts {contrast1}→{contrast2}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).unsqueeze(0).float()
