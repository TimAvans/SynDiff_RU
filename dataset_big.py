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

from nibabel import orientations

def _reorient_to_RAS(data, affine):
    orig_ornt = orientations.io_orientation(affine)
    ras_ornt = orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = orientations.ornt_transform(orig_ornt, ras_ornt)
    return orientations.apply_orientation(data, transform), transform

def _world_z_slices(affine, shape, orientation):
    # Bepaal hoe oriëntatie en affine samen de Z-as beïnvloeden
    inv_transform = orientations.inv_ornt_aff(orientation, shape)
    coords = [affine @ (inv_transform @ np.array([0, 0, z, 1])) for z in range(shape[2])]
    return np.array([c[2] for c in coords])  # Z-positie in wereldruimte


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
            print(f"\n>> Subject {subject}")
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

            print(f"T1 shape: {vol1.shape}, T2 shape: {vol2.shape}")
            print(f"T1 orient: {orientations.ornt2axcodes(orientations.io_orientation(img1.affine))}")
            print(f"T2 orient: {orientations.ornt2axcodes(orientations.io_orientation(img2.affine))}")

            z1_world = _world_z_slices(img1.affine, img1.shape, t1_ornt)
            z2_world = _world_z_slices(img2.affine, img2.shape, t2_ornt)

            target_z = np.median(z1_world)

            z1_idx = np.argmin(np.abs(z1_world - target_z))
            z2_idx = np.argmin(np.abs(z2_world - target_z))

            print(f"World-Z posities: T1[{z1_idx}] = {z1_world[z1_idx]:.2f}, T2[{z2_idx}] = {z2_world[z2_idx]:.2f}")

            sl1 = _crop_or_pad(vol1[:, :, z1_idx], size=size).astype(np.float32)
            sl2 = _crop_or_pad(vol2[:, :, z2_idx], size=size).astype(np.float32)

            # Toon voorbeeldplaatjes van eerste 3
            if count < 3:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6,3))
                plt.subplot(1,2,1)
                plt.imshow(sl1, cmap='gray')
                plt.title(f'{subject} - T1')
                plt.subplot(1,2,2)
                plt.imshow(sl2, cmap='gray')
                plt.title(f'{subject} - T2')
                plt.show()

            x_list.append(sl1)
            y_list.append(sl2)
            count += 1
            if max_slices is not None and count >= max_slices:
                break

    print(f"\n✅ Total matched slices: {len(x_list)}")

    if len(x_list) == 0:
        raise RuntimeError("No matching slices found.")

    x = np.stack(x_list)[:, None, :, :]
    y = np.stack(y_list)[:, None, :, :]

    for arr in [x, y]:
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr[:] = (arr - 0.5) / 0.5

    return torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

