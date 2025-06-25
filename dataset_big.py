import os
import tarfile
import gzip
import io
import numpy as np
import torch.utils.data

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - nibabel may be missing
    nib = None


def _crop_or_pad(img, size=256):
    """Center crop or pad a 2D image to the desired size."""
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


def LoadDataSetBig(root_dir, contrast, max_slices=None, normalize=True, size=256):
    """Load slices from ``IXI_{contrast}.tar`` located in ``root_dir``.

    Only the middle 50% of slices from every volume are returned.
    ``max_slices`` limits the total number of slices loaded.
    """

    if nib is None:
        raise ImportError("nibabel is required for dataset_big")

    tar_path = os.path.join(root_dir, f"IXI_{contrast}.tar")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(tar_path)

    data = []
    count = 0
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith((".nii", ".nii.gz"))]
        members.sort(key=lambda m: m.name)

        for m in members:
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

    if not data:
        raise RuntimeError(f"No data found in {tar_path}")

    data = np.stack(data, axis=0)
    data = data[:, None, :, :]

    if normalize:
        data = data - data.min()
        if data.max() != 0:
            data = data / data.max()
        data = (data - 0.5) / 0.5

    return data


def CreateDatasetSynthesis(phase, input_path, contrast1="T1", contrast2="T2", max_slices=None, size=256):
    """Return a dataset of paired slices loaded from IXI ``.tar`` archives."""

    data1 = LoadDataSetBig(input_path, contrast1, max_slices=max_slices, size=size)
    data2 = LoadDataSetBig(input_path, contrast2, max_slices=max_slices, size=size)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data1), torch.from_numpy(data2))
    return dataset