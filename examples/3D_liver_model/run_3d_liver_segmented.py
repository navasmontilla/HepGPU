import os
import numpy as np
import SimpleITK as sitk

from scipy.ndimage import zoom

from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png, export_zarr_to_2d_snapshot


# ==========================================
# Input segmentation
# ==========================================

segmentation_file = "Segmentation_1.nrrd"


# ==========================================
# Grid
# ==========================================

nx, ny, nz = 64, 64, 36
Lx, Ly, Lz = 1.0, 1.0, 1.0


# ==========================================
# Read segmentation
# ==========================================

img = sitk.ReadImage(segmentation_file)
arr = sitk.GetArrayFromImage(img)   # expected shape: (z, y, x)

labels_zyx = arr

print("Segmentation shape:", labels_zyx.shape)
print("Unique labels:", np.unique(labels_zyx))


# ==========================================
# Build masks in native segmentation space
# ==========================================

ID_VEIN = 1
ID_LOBES = [5, 6, 7, 8, 9, 10, 11, 12]

mask_vein_zyx = (labels_zyx == ID_VEIN)

mask_left_lobe_2_zyx     = (labels_zyx == ID_LOBES[0])
mask_right_lobe_8_zyx    = (labels_zyx == ID_LOBES[1])
mask_central_lobe_4a_zyx = (labels_zyx == ID_LOBES[2])
mask_left_lobe_3_zyx     = (labels_zyx == ID_LOBES[3])
mask_central_lobe_4b_zyx = (labels_zyx == ID_LOBES[4])
mask_post_lobe_7_zyx     = (labels_zyx == ID_LOBES[5])
mask_right_lobe_5_zyx    = (labels_zyx == ID_LOBES[6])
mask_post_lobe_6_zyx     = (labels_zyx == ID_LOBES[7])


# ==========================================
# Convert masks to (x, y, z)
# ==========================================

mask_vein = np.transpose(mask_vein_zyx, (2, 1, 0))

mask_left_lobe_2     = np.transpose(mask_left_lobe_2_zyx,     (2, 1, 0))
mask_right_lobe_8    = np.transpose(mask_right_lobe_8_zyx,    (2, 1, 0))
mask_central_lobe_4a = np.transpose(mask_central_lobe_4a_zyx, (2, 1, 0))
mask_left_lobe_3     = np.transpose(mask_left_lobe_3_zyx,     (2, 1, 0))
mask_central_lobe_4b = np.transpose(mask_central_lobe_4b_zyx, (2, 1, 0))
mask_post_lobe_7     = np.transpose(mask_post_lobe_7_zyx,     (2, 1, 0))
mask_right_lobe_5    = np.transpose(mask_right_lobe_5_zyx,    (2, 1, 0))
mask_post_lobe_6     = np.transpose(mask_post_lobe_6_zyx,     (2, 1, 0))


# ==========================================
# Resample masks to simulation grid
# ==========================================

sx, sy, sz = mask_vein.shape
zx, zy, zz = nx / sx, ny / sy, nz / sz

def resample_mask(mask):
    return zoom(mask.astype(np.uint8), (zx, zy, zz), order=0).astype(bool)

mask_vein = resample_mask(mask_vein)

mask_left_lobe_2     = resample_mask(mask_left_lobe_2)
mask_right_lobe_8    = resample_mask(mask_right_lobe_8)
mask_central_lobe_4a = resample_mask(mask_central_lobe_4a)
mask_left_lobe_3     = resample_mask(mask_left_lobe_3)
mask_central_lobe_4b = resample_mask(mask_central_lobe_4b)
mask_post_lobe_7     = resample_mask(mask_post_lobe_7)
mask_right_lobe_5    = resample_mask(mask_right_lobe_5)
mask_post_lobe_6     = resample_mask(mask_post_lobe_6)


# ==========================================
# Resolve overlaps
# ==========================================

lobes = [
    mask_left_lobe_2,
    mask_right_lobe_8,
    mask_central_lobe_4a,
    mask_left_lobe_3,
    mask_central_lobe_4b,
    mask_post_lobe_7,
    mask_right_lobe_5,
    mask_post_lobe_6,
]

lobes = [m & (~mask_vein) for m in lobes]

occupied = mask_vein.copy()
lobes_clean = []

for m in lobes:
    mc = m & (~occupied)
    lobes_clean.append(mc)
    occupied |= mc

(
    mask_left_lobe_2,
    mask_right_lobe_8,
    mask_central_lobe_4a,
    mask_left_lobe_3,
    mask_central_lobe_4b,
    mask_post_lobe_7,
    mask_right_lobe_5,
    mask_post_lobe_6,
) = lobes_clean

mask_liver = np.logical_or.reduce(lobes_clean)
mask_background = ~occupied

print("Vein voxels:", mask_vein.sum())
print("Liver voxels:", mask_liver.sum())
print("Background voxels:", mask_background.sum())


# ==========================================
# Define masks for HepGPU
# ==========================================

background_values = (0.0, 0.0, 0.0, 0.0)

masks = [
    (mask_background, background_values),

    (mask_left_lobe_2,     (0.57, 0.85, 0.45, 0.48)),
    (mask_right_lobe_8,    (0.63, 0.90, 0.55, 0.52)),
    (mask_central_lobe_4a, (0.60, 0.88, 0.48, 0.50)),
    (mask_left_lobe_3,     (0.57, 0.85, 0.45, 0.48)),
    (mask_central_lobe_4b, (0.58, 0.87, 0.46, 0.49)),
    (mask_post_lobe_7,     (0.63, 0.90, 0.55, 0.52)),
    (mask_right_lobe_5,    (0.63, 0.90, 0.55, 0.52)),
    (mask_post_lobe_6,     (0.63, 0.90, 0.55, 0.52)),
]


# ==========================================
# Model parameters
# ==========================================

params = {
    "a1": 1.0,
    "C1": 1.0,
    "epsilon": 0.05,
    "kappa": 0.01,
    "a5": 0.05,
    "a2h": 2.0,
    "Cth": 8.0,
    "a6h": 0.2,
    "a2c": 2.0,
    "Ctc": 15.0,
    "a6c": 0.2,
    "a3": 0.8,
    "a_nd": 0.6,
}

out_name = "output_3d_liver_segmented.zarr"
results_dir = "results_3d_liver_segmented"


# ==========================================
# Run simulation
# ==========================================

if __name__ == "__main__":
    run(
        use_gpu=False,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,

        nx=nx,
        ny=ny,
        nz=nz,

        tf=5,
        td=1,

        sigma=0.15,
        params=params,
        mask_inflow=mask_vein,
        CI_values=(0.1, 0.0, 0.0, 0.0),
        masks=masks,
        out_name=out_name,
    )

    export_zarr_to_vtk(
        zarr_file=out_name,
        output_dir=os.path.join(results_dir, "vtk"),
    )

    export_zarr_to_png(
        zarr_file=out_name,
        output_dir=os.path.join(results_dir, "plots"),
    )

    export_zarr_to_2d_snapshot(
        zarr_file=out_name,
        output_path=os.path.join(results_dir, "spatial_snapshot_step_400.png"),
        step=10,
    )

    print("3D segmented-liver simulation completed successfully.")
