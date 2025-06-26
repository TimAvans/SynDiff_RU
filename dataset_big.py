import os
import tarfile
import gzip
import re
import numpy as np
import torch.utils.data

try:
    import nibabel as nib
    from nibabel import orientations
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

def _is_axial(affine):
    """Check if image has axial orientation ('R', 'A', 'S')."""
    orientation = orientations.ornt2axcodes(orientations.io_orientation(affine))
    return orientation == ('R', 'A', 'S')

def CreateDatasetSynthesis(phase, input_path, contrast1="T1", contrast2="T2", max_slices=None, size=256):
    tar_path1 = os.path.join(input_path, f"IXI_{contrast1}.tar")
    tar_path2 = os.path.join(input_path, f"IXI_{contrast2}.tar")

    if nib is None:
        raise ImportError("nibabel is required")

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

            # Skip non-axial scans
            if not (_is_axial(img1.affine) and _is_axial(img2.affine)):
                continue

            vol1 = img1.get_fdata()
            vol2 = img2.get_fdata()

            min_slices = min(vol1.shape[2], vol2.shape[2])
            for z in range(min_slices):
                sl1 = _crop_or_pad(vol1[:, :, z], size=size).astype(np.float32)
                sl2 = _crop_or_pad(vol2[:, :, z], size=size).astype(np.float32)

                x_list.append(sl1)
                y_list.append(sl2)
                count += 1
                if max_slices is not None and count >= max_slices:
                    break
            if max_slices is not None and count >= max_slices:
                break

    x = np.stack(x_list)[:, None, :, :]
    y = np.stack(y_list)[:, None, :, :]

    for arr in [x, y]:
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr[:] = (arr - 0.5) / 0.5

    return torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
