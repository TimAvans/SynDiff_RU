import torch.utils.data
import numpy as np
import nibabel as nib
import os
import tarfile
import tempfile
import re

class SynDiffDataset(torch.utils.data.Dataset):
    def __init__(self, phase, input_path, contrast1='T1', contrast2='T2'):
        self.phase = phase
        self.input_path = input_path
        self.tar_t1 = tarfile.open(os.path.join(input_path, f'IXI-{contrast1}.tar'), 'r')
        self.tar_t2 = tarfile.open(os.path.join(input_path, f'IXI-{contrast2}.tar'), 'r')
        self.contrast1_files = [f for f in self.tar_t1.getnames() if f.endswith('.nii.gz') and contrast1 in os.path.basename(f)]
        self.contrast2_files = [f for f in self.tar_t2.getnames() if f.endswith('.nii.gz') and contrast2 in os.path.basename(f)]
        self.padding = True
        self.Norm = True

        # Build subject-to-file mapping
        def extract_subject_id(filename):
            # Example: IXI123-T1.nii.gz -> IXI123
            match = re.match(r'(IXI\d+)', os.path.basename(filename))
            return match.group(1) if match else None

        t1_subjects = {extract_subject_id(f): f for f in self.contrast1_files}
        t2_subjects = {extract_subject_id(f): f for f in self.contrast2_files}
        self.common_subjects = sorted(set(t1_subjects.keys()) & set(t2_subjects.keys()))

        # Build list of (subject, slice_idx) pairs (middle 50% only)
        self.pairs = []
        for subj in self.common_subjects:
            t1_file = t1_subjects[subj]
            t2_file = t2_subjects[subj]
            # Load both volumes to determine min number of slices
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp1, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp2:
                tmp1.write(self.tar_t1.extractfile(t1_file).read())
                tmp2.write(self.tar_t2.extractfile(t2_file).read())
                t1_img = nib.load(tmp1.name)
                t2_img = nib.load(tmp2.name)
                os.unlink(tmp1.name)
                os.unlink(tmp2.name)
            num_slices = min(t1_img.shape[2], t2_img.shape[2])
            # Only use the middle 50% of slices
            start = num_slices // 4
            end = start + num_slices // 2
            for z in range(start, end):
                self.pairs.append((subj, z))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        subj, z = self.pairs[idx]
        t1_file = [f for f in self.contrast1_files if subj in f][0]
        t2_file = [f for f in self.contrast2_files if subj in f][0]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp2:
            tmp1.write(self.tar_t1.extractfile(t1_file).read())
            tmp2.write(self.tar_t2.extractfile(t2_file).read())
            # Load both images
            t1_img = nib.load(tmp1.name)
            t2_img = nib.load(tmp2.name)
            # Resample T2 to T1's grid
            t2_img_resampled = nib.processing.resample_from_to(t2_img, t1_img)
            # Reorient both to RAS+
            t1_data = self._reorient_to_ras(t1_img)
            t2_data = self._reorient_to_ras(t2_img_resampled)
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)
        # Get the z-th slice
        x = t1_data[z % t1_data.shape[0]]
        y = t2_data[z % t2_data.shape[0]]
        return x, y

    def _reorient_to_ras(self, img):
        orig_ornt = nib.orientations.io_orientation(img.affine)
        target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
        data = img.get_fdata()
        data = nib.orientations.apply_orientation(data, transform)
        slices = []
        for z in range(data.shape[2]):
            slice_data = data[:, :, z]
            slice_data = np.expand_dims(slice_data, axis=0)  # [1, H, W]
            slices.append(slice_data)
        data = np.stack(slices, axis=0)  # [num_slices, 1, H, W]
        data = data.astype(np.float32)
        if self.padding:
            pad_x = int((256 - data.shape[2]) / 2)
            pad_y = int((256 - data.shape[3]) / 2)
            if pad_x > 0 or pad_y > 0:
                data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))
        if self.Norm:
            data = (data - np.mean(data)) / np.std(data)
            data = data * 2 - 1
        return data


    # Load a single NIfTI file and extract slices
    def LoadDataSet(self, file_path):
        img = nib.load(file_path)
        # Reorient to RAS+ (axial) orientation
        orig_ornt = nib.orientations.io_orientation(img.affine)
        target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
        data = img.get_fdata()
        data = nib.orientations.apply_orientation(data, transform)
        slices = []
        for z in range(data.shape[2]):
            slice_data = data[:, :, z]
            slice_data = np.expand_dims(slice_data, axis=0)  # [1, H, W]
            slices.append(slice_data)
        data = np.stack(slices, axis=0)  # [num_slices, 1, H, W]
        data = data.astype(np.float32)
        if self.padding:
            pad_x = int((256 - data.shape[2]) / 2)
            pad_y = int((256 - data.shape[3]) / 2)
            if pad_x > 0 or pad_y > 0:
                data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))
        if self.Norm:
            data = (data - np.mean(data)) / np.std(data)
            data = data * 2 - 1
        return data

    def __del__(self):
        if hasattr(self, 'tar_t1'):
            self.tar_t1.close()
        if hasattr(self, 'tar_t2'):
            self.tar_t2.close()

def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    return SynDiffDataset(phase, input_path, contrast1, contrast2)