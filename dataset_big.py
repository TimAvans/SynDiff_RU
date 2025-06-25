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


from nibabel import orientations

def _reorient_to_RAS(data, affine):
    orig_ornt = orientations.io_orientation(affine)
    ras_ornt = orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = orientations.ornt_transform(orig_ornt, ras_ornt)
    return orientations.apply_orientation(data, transform)


from nibabel import orientations
import numpy.linalg as la

def _reorient_to_RAS(data, affine):
    orig_ornt = orientations.io_orientation(affine)
    ras_ornt = orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = orientations.ornt_transform(orig_ornt, ras_ornt)
    return orientations.apply_orientation(data, transform), transform


def _world_z_slices(affine, shape, transform):
    # Bereken fysieke z-positie per slice
    inv_transform = orientations.inv_ornt_aff(transform, shape)
    coords = [np.dot(affine, inv_transform @ np.array([0, 0, z, 1])) for z in range(shape[2])]
    return np.array([c[2] for c in coords])  # pak alleen de Z-coördinaat in wereldruimte

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

            vol1, t1_ornt = _reorient_to_RAS(img1.get_fdata(), img1.affine)
            vol2, t2_ornt = _reorient_to_RAS(img2.get_fdata(), img2.affine)

            z1_world = _world_z_slices(img1.affine, img1.shape, t1_ornt)
            z2_world = _world_z_slices(img2.affine, img2.shape, t2_ornt)

            target_z = np.median(z1_world)

            # Zoek dichtstbijzijnde slice in beide volumes
            z1_idx = np.argmin(np.abs(z1_world - target_z))
            z2_idx = np.argmin(np.abs(z2_world - target_z))

            sl1 = _crop_or_pad(vol1[:, :, z1_idx], size=size).astype(np.float32)
            sl2 = _crop_or_pad(vol2[:, :, z2_idx], size=size).astype(np.float32)

            x_list.append(sl1)
            y_list.append(sl2)
            count += 1
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

