import os
import tarfile
import gzip
import io
import re
import numpy as np
import torch.utils.data

try:
    import nibabel as nib
except ImportError:
    nib = None


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


def _load_from_tar(tar_path, selected_members, max_slices=None, normalize=True, size=256):
    if nib is None:
        raise ImportError("nibabel is required")

    data = []
    count = 0
    with tarfile.open(tar_path, "r") as tar:
        for m in selected_members:
            fobj = tar.extractfile(m)
            if fobj is None:
                continue
            file_bytes = fobj.read()
            if m.name.endswith(".gz"):
                file_bytes = gzip.decompress(file_bytes)

            img = nib.Nifti1Image.from_bytes(file_bytes)
            volume = img.get_fdata()
            z0 = volume.shape[2] // 4
            z1 = volume.shape[2] - z0

            for z in range(z0, z1):
                sl = volume[:, :, z]
                sl = _crop_or_pad(sl, size=size)
                data.append(sl.astype(np.float32))
                count += 1
                if max_slices is not None and count >= max_slices:
                    break
            if max_slices is not None and count >= max_slices:
                break

    data = np.stack(data, axis=0)
    data = data[:, None, :, :]

    if normalize:
        data = data - data.min()
        if data.max() != 0:
            data = data / data.max()
        data = (data - 0.5) / 0.5

    return data


def CreateDatasetSynthesis(phase, input_path, contrast1="T1", contrast2="T2", max_slices=None, size=256):
    tar_path1 = os.path.join(input_path, f"IXI_{contrast1}.tar")
    tar_path2 = os.path.join(input_path, f"IXI_{contrast2}.tar")

    with tarfile.open(tar_path1, "r") as tar1, tarfile.open(tar_path2, "r") as tar2:
        members1 = [m for m in tar1.getmembers() if m.isfile()]
        members2 = [m for m in tar2.getmembers() if m.isfile()]

        subjects1 = { _extract_subject_id(m.name): m for m in members1 if _extract_subject_id(m.name) }
        subjects2 = { _extract_subject_id(m.name): m for m in members2 if _extract_subject_id(m.name) }

        common_subjects = sorted(set(subjects1.keys()) & set(subjects2.keys()))

        selected1 = [subjects1[s] for s in common_subjects]
        selected2 = [subjects2[s] for s in common_subjects]

    data1 = _load_from_tar(tar_path1, selected1, max_slices=max_slices, size=size)
    data2 = _load_from_tar(tar_path2, selected2, max_slices=max_slices, size=size)

    # truncate to same number of slices
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    return torch.utils.data.TensorDataset(torch.from_numpy(data1), torch.from_numpy(data2))
