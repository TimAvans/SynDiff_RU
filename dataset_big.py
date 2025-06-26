import os
import gzip
import tarfile
import re
import numpy as np
import torch
import torch.utils.data
import nibabel as nib
from skimage.transform import resize


def _extract_subject_id(name):
    match = re.match(r"(IXI\d+)", os.path.basename(name))
    return match.group(1) if match else None


def _normalize_volume(volume):
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)
    return volume


def _resize_and_crop(volume, target_size=256):
    # volume: (H, W, Z)
    resized = resize(volume, (target_size, target_size, volume.shape[2]), preserve_range=True)
    return resized


def CreateDatasetSynthesis(phase, input_path, contrast1="T1", contrast2="T2", max_slices=None, size=256):
    tar_path1 = os.path.join(input_path, f"IXI_{contrast1}.tar")
    tar_path2 = os.path.join(input_path, f"IXI_{contrast2}.tar")

    with tarfile.open(tar_path1, "r") as tar1, tarfile.open(tar_path2, "r") as tar2:
        members1 = [m for m in tar1.getmembers() if m.isfile()]
        members2 = [m for m in tar2.getmembers() if m.isfile()]

        subjects1 = {_extract_subject_id(m.name): m for m in members1 if _extract_subject_id(m.name)}
        subjects2 = {_extract_subject_id(m.name): m for m in members2 if _extract_subject_id(m.name)}

        common_subjects = sorted(set(subjects1.keys()) & set(subjects2.keys()))

        x_list, y_list = [], []
        count = 0

        for subject in common_subjects:
            m1 = subjects1[subject]
            m2 = subjects2[subject]

            fobj1 = tar1.extractfile(m1)
            fobj2 = tar2.extractfile(m2)
            if fobj1 is None or fobj2 is None:
                continue

            img_bytes1 = gzip.decompress(fobj1.read()) if m1.name.endswith(".gz") else fobj1.read()
            img_bytes2 = gzip.decompress(fobj2.read()) if m2.name.endswith(".gz") else fobj2.read()

            vol1 = nib.Nifti1Image.from_bytes(img_bytes1).get_fdata()
            vol2 = nib.Nifti1Image.from_bytes(img_bytes2).get_fdata()

            vol1 = _resize_and_crop(_normalize_volume(vol1), target_size=size)
            vol2 = _resize_and_crop(_normalize_volume(vol2), target_size=size)

            num_slices = min(vol1.shape[2], vol2.shape[2])

            for i in range(num_slices):
                x_slice = vol1[:, :, i].astype(np.float32)
                y_slice = vol2[:, :, i].astype(np.float32)

                x_list.append(x_slice)
                y_list.append(y_slice)
                count += 1
                if max_slices is not None and count >= max_slices:
                    break
            if max_slices is not None and count >= max_slices:
                break

    x = np.stack(x_list)[:, None, :, :]  # shape: [N, 1, H, W]
    y = np.stack(y_list)[:, None, :, :]

    # Scale to [-1, 1]
    for arr in [x, y]:
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr[:] = (arr - 0.5) / 0.5

    return torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
