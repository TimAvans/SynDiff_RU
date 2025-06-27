import os, tarfile, gzip, re, numpy as np, torch.utils.data
import nibabel as nib
from nibabel import orientations
import SimpleITK as sitk
import tempfile

def _crop_or_pad(img, size=256):
    h, w = img.shape
    img = img[:size, :size] if h >= size and w >= size else np.pad(img, ((0, max(0, size-h)), (0, max(0, size-w))))
    return img[:size, :size]

def _extract_subject_id(name):
    match = re.match(r"(IXI\d+)", os.path.basename(name))
    return match.group(1) if match else None

def _reorient_to_RAS(data, affine):
    ornt = orientations.io_orientation(affine)
    ras_ornt = orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = orientations.ornt_transform(ornt, ras_ornt)
    return orientations.apply_orientation(data, transform)

def _register_affine(fixed, moving):
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))
    transform = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 100)
    registration.SetInterpolator(sitk.sitkLinear)
    final_transform = registration.Execute(fixed_sitk, moving_sitk)
    aligned = sitk.Resample(moving_sitk, fixed_sitk, final_transform, sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())
    return sitk.GetArrayFromImage(aligned)

def CreateDatasetSynthesisAligned(phase, input_path, contrast1="T1", contrast2="T2", max_slices=None, size=256):
    tar_path1 = os.path.join(input_path, f"IXI_{contrast1}.tar")
    tar_path2 = os.path.join(input_path, f"IXI_{contrast2}.tar")

    with tarfile.open(tar_path1, "r") as tar1, tarfile.open(tar_path2, "r") as tar2:
        members1 = [m for m in tar1.getmembers() if m.isfile()]
        members2 = [m for m in tar2.getmembers() if m.isfile()]

        subjects1 = { _extract_subject_id(m.name): m for m in members1 if _extract_subject_id(m.name) }
        subjects2 = { _extract_subject_id(m.name): m for m in members2 if _extract_subject_id(m.name) }

        common_subjects = sorted(set(subjects1) & set(subjects2))

        x_list, y_list, count = [], [], 0

        for subject in common_subjects:
            with tar1.extractfile(subjects1[subject]) as f1, tar2.extractfile(subjects2[subject]) as f2:
                b1, b2 = gzip.decompress(f1.read()) if subjects1[subject].name.endswith(".gz") else f1.read(), gzip.decompress(f2.read()) if subjects2[subject].name.endswith(".gz") else f2.read()
                img1, img2 = nib.Nifti1Image.from_bytes(b1), nib.Nifti1Image.from_bytes(b2)
                vol1, vol2 = _reorient_to_RAS(img1.get_fdata(), img1.affine), _reorient_to_RAS(img2.get_fdata(), img2.affine)
                vol2_aligned = _register_affine(vol1, vol2)

                for z in range(min(vol1.shape[2], vol2_aligned.shape[2]) // 4, min(vol1.shape[2], vol2_aligned.shape[2]) * 3 // 4):
                    x_list.append(_crop_or_pad(vol1[:, :, z], size).astype(np.float32))
                    y_list.append(_crop_or_pad(vol2_aligned[:, :, z], size).astype(np.float32))
                    count += 1
                    if max_slices is not None and count >= max_slices:
                        break
                if max_slices is not None and count >= max_slices:
                    break

    if not x_list: raise ValueError("No aligned slices found.")
    x = np.stack(x_list)[:, None, :, :]
    y = np.stack(y_list)[:, None, :, :]

    for arr in [x, y]:
        arr -= arr.min()
        if arr.max() > 0: arr /= arr.max()
        arr[:] = (arr - 0.5) / 0.5

    return torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
