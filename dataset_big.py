import os
import tarfile
import gzip
import re
import numpy as np
import torch.utils.data

import nibabel as nib
from nibabel import orientations

def _crop_or_pad(img, size=256):
    h, w = img.shape
    if h > size:
        start = (h - size) // 2
        img = img[start:start + size, :]
        h = size
    if w > size:
        start = (w - size) // 2
        img = img[:, start:start + size]
        w = size
    pad_h = size - h
    pad_w = size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)))
    return img

def _extract_subject_id(name):
    match = re.match(r"(IXI\d+)", os.path.basename(name))
    return match.group(1) if match else None

def _reorient_to_RAS(data, affine):
    ornt = orientations.io_orientation(affine)
    ras_ornt = orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = orientations.ornt_transform(ornt, ras_ornt)
    return orientations.apply_orientation(data, transform)

def CreateDatasetSynthesis(phase, input_path, contrast1="T1", contrast2="T2", max_slices=None, size=256):
    tar_path1 = os.path.join(input_path, f"IXI_{contrast1}.tar")
    tar_path2 = os.path.join(input_path, f"IXI_{contrast2}.tar")

    with tarfile.open(tar_path1, "r") as tar1, tarfile.open(tar_path2, "r") as tar2:
        members1 = [m for m in tar1.getmembers() if m.isfile()]
        members2 = [m for m in tar2.getmembers() if m.isfile()]

        subjects1 = { _extract_subject_id(m.name): m for m in members1 if _extract_subject_id(m.name) }
        subjects2 = { _extract_subject_id(m.name): m for m in members2 if _extract_subject_id(m.name) }

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
            bytes1 = fobj1.read()
            bytes2 = fobj2.read()
            if m1.name.endswith(".gz"):
                bytes1 = gzip.decompress(bytes1)
            if m2.name.endswith(".gz"):
                bytes2 = gzip.decompress(bytes2)

            img1 = nib.Nifti1Image.from_bytes(bytes1)
            img2 = nib.Nifti1Image.from_bytes(bytes2)

            vol1 = _reorient_to_RAS(img1.get_fdata(), img1.affine)
            vol2 = _reorient_to_RAS(img2.get_fdata(), img2.affine)

            min_slices = min(vol1.shape[2], vol2.shape[2])
            start = min_slices // 4
            end = start + (min_slices // 2)
            for z in range(start, end):
                sl1 = _crop_or_pad(vol1[:, :, z], size=size).astype(np.float32)
                sl2 = _crop_or_pad(vol2[:, :, z], size=size).astype(np.float32)

                x_list.append(sl1)
                y_list.append(sl2)
                count += 1
                if max_slices is not None and count >= max_slices:
                    break
            if max_slices is not None and count >= max_slices:
                break

    if not x_list:
        raise ValueError("No usable slices found. Check data or orientation handling.")

    x = np.stack(x_list)[:, None, :, :]
    y = np.stack(y_list)[:, None, :, :]

    for arr in [x, y]:
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr[:] = (arr - 0.5) / 0.5

    return torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
