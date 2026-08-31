import os
import numpy as np
import SimpleITK as sitk
import zarr
import matplotlib.pyplot as plt

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


# ==========================================
# Read segmentation
# ==========================================

img = sitk.ReadImage(segmentation_file)
arr = sitk.GetArrayFromImage(img)   # expected shape: (z, y, x)

labels_zyx = arr

print("Original segmentation shape:", labels_zyx.shape)
print("Original unique labels:", np.unique(labels_zyx))


# ==========================================
# Labels
# ==========================================

# .nrrd file tags
ID_VEIN = 1
ID_LOBES = [5, 6, 7, 8, 9, 10, 11, 12]


# ==========================================
# Crop segmentation to liver bounding box
# ==========================================

# mask_crop_zyx: boolean mask that is worth True where there is liver or vein
mask_crop_zyx = np.zeros_like(labels_zyx, dtype=bool) 
mask_crop_zyx |= (labels_zyx == ID_VEIN)
for label in ID_LOBES:
    mask_crop_zyx |= (labels_zyx == label)

z_idx, y_idx, x_idx = np.where(mask_crop_zyx) # coordinates of all useful voxels 

# minimal box that contains the liver
zmin, zmax = z_idx.min(), z_idx.max()
ymin, ymax = y_idx.min(), y_idx.max()
xmin, xmax = x_idx.min(), x_idx.max()

margin = 5 # margin to avoid losing edges

zmin = max(zmin - margin, 0)
ymin = max(ymin - margin, 0)
xmin = max(xmin - margin, 0)

zmax = min(zmax + margin, labels_zyx.shape[0] - 1)
ymax = min(ymax + margin, labels_zyx.shape[1] - 1)
xmax = min(xmax + margin, labels_zyx.shape[2] - 1)

labels_zyx = labels_zyx[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] # reduce the volume

print("Cropped segmentation shape:", labels_zyx.shape)
print("Cropped unique labels:", np.unique(labels_zyx))


# ==========================================
# Real physical size of cropped domain
# ==========================================

spacing = img.GetSpacing()  # (x, y, z) in mm

nz_crop, ny_crop, nx_crop = labels_zyx.shape # number of voxels in (z, y, x)

Lx = nx_crop * spacing[0] / 1000.0 # in m
Ly = ny_crop * spacing[1] / 1000.0
Lz = nz_crop * spacing[2] / 1000.0

print("Real cropped domain size:")
print(f"Lx = {Lx:.4f} m")
print(f"Ly = {Ly:.4f} m")
print(f"Lz = {Lz:.4f} m")

# ==========================================
# Build masks in native segmentation space
# ==========================================

mask_vein_zyx = (labels_zyx == ID_VEIN) # voxels that contain the tag 1 (ID_VEIN)

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

def resample_mask_nn(mask, new_shape):
    sx, sy, sz = mask.shape
    nx_new, ny_new, nz_new = new_shape

    ix = np.round(np.linspace(0, sx - 1, nx_new)).astype(int)
    iy = np.round(np.linspace(0, sy - 1, ny_new)).astype(int)
    iz = np.round(np.linspace(0, sz - 1, nz_new)).astype(int)

    return mask[np.ix_(ix, iy, iz)].astype(bool)

mask_vein = resample_mask_nn(mask_vein, (nx, ny, nz))

mask_left_lobe_2     = resample_mask_nn(mask_left_lobe_2,     (nx, ny, nz))
mask_right_lobe_8    = resample_mask_nn(mask_right_lobe_8,    (nx, ny, nz))
mask_central_lobe_4a = resample_mask_nn(mask_central_lobe_4a, (nx, ny, nz))
mask_left_lobe_3     = resample_mask_nn(mask_left_lobe_3,     (nx, ny, nz))
mask_central_lobe_4b = resample_mask_nn(mask_central_lobe_4b, (nx, ny, nz))
mask_post_lobe_7     = resample_mask_nn(mask_post_lobe_7,     (nx, ny, nz))
mask_right_lobe_5    = resample_mask_nn(mask_right_lobe_5,    (nx, ny, nz))
mask_post_lobe_6     = resample_mask_nn(mask_post_lobe_6,     (nx, ny, nz))


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
mask_domain = mask_liver | mask_vein

print("Vein voxels:", mask_vein.sum())
print("Liver voxels:", mask_liver.sum())
print("Background voxels:", mask_background.sum())
print("Domain voxels (liver + vein):", mask_domain.sum())

# ==========================================
# Build fibrosis barriers
# ==========================================

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# lower elongated barrier across lobe 5 and 6 
mask_band_lower_5_and_6 = ( 
    (X > 0.0) & (X < 0.2644) & # atraviesa lateralmente 
    (Y > 0.0) & (Y < 0.14) & # más fina en Y => completa con 0.2238 (con 0.14 salía como la anterior)
    (Z > 0.162) & (Z < 0.171) # MUY fina en Z 0.009 (0.153-0.162)
)

mask_band_6_and_7_new = (
    (X > 0.000) & (X < 0.1130) &      # cierra la apertura lateral (0.1200)
    (Y > 0.175) & (Y < 0.185) &      # también más arriba
    (Z > 0.000) & (Z < 0.3150)
)

# thin posterior barrier in the right tip
mask_band_2_and_3 = (
    (X > 0.168) & (X < 0.176) &     # más fina 0.008 diferencia
    (Y > 0.015) & (Y < 0.2238) &
    (Z > 0.0) & (Z < 0.315)         
)

mask_band_2_and_3_new = (
    (X > 0.192) & (X < 0.2) &     # más fina
    (Y > 0.0) & (Y < 0.1) & #0.095
    (Z > 0.0) & (Z < 0.315)         
)

mask_fibrosis_lower_5_and_6 = mask_band_lower_5_and_6 & (mask_right_lobe_5 | mask_post_lobe_6)
mask_fibrosis_6_and_7_new = mask_band_6_and_7_new & (mask_post_lobe_6 | mask_post_lobe_7)
mask_fibrosis_2_and_3 = mask_band_2_and_3 & (mask_left_lobe_2 | mask_left_lobe_3)
mask_fibrosis_2_and_3_new = mask_band_2_and_3_new & (mask_left_lobe_2 | mask_left_lobe_3)

mask_fibrosis = mask_fibrosis_lower_5_and_6 | mask_fibrosis_2_and_3 | mask_fibrosis_6_and_7_new | mask_fibrosis_2_and_3_new

print("Fibrosis 6 and 7 voxels new:", mask_fibrosis_6_and_7_new.sum())
print("Fibrosis lower 5 and 6 voxels:", mask_fibrosis_lower_5_and_6.sum())
print("Fibrosis 2 and 3 voxels:", mask_fibrosis_2_and_3.sum())
print("Fibrosis 2 and 3 voxels new:", mask_fibrosis_2_and_3_new.sum())
print("Fibrosis total voxels:", mask_fibrosis.sum())

# ==========================================
# Define masks for HepGPU
# ==========================================

background_values = (0.0, 0.0, 0.0, 0.0)

masks = [
    (mask_background, background_values),

    # (d1, dTh, dTc, d3)
    (mask_left_lobe_2,     (0.006, 0.08, 0.05, 0.04)),
    (mask_right_lobe_8,    (0.008, 0.09, 0.07, 0.05)),
    (mask_central_lobe_4a, (0.007, 0.085, 0.06, 0.045)),
    (mask_left_lobe_3,     (0.006, 0.08, 0.05, 0.04)),
    (mask_central_lobe_4b, (0.007, 0.085, 0.06, 0.045)),
    (mask_post_lobe_7,     (0.008, 0.09, 0.07, 0.05)),
    (mask_right_lobe_5,    (0.008, 0.09, 0.07, 0.05)),
    (mask_post_lobe_6,     (0.008, 0.09, 0.07, 0.05)),

    # barriers
    (mask_fibrosis,        (0.0, 0.0, 0.0, 0.0)),
]


# ==========================================
# Model parameters
# ==========================================

params = {

    "a1": 1.0,
    "C1": 1.0,
    "epsilon": 0.05,
    "kappa": 0.05,

    "a5": 0.036,

    "a2h": 14.0,
    "Cth": 8.0,
    "a6h": 0.2,

    "a2c": 8.0,
    "Ctc": 15.0,
    "a6c": 0.03,

    "a3": 0.6,
    "a_nd": 1.0,
}

out_name = "output_3d_liver_segmented_barrier_Prueba2_t100_L_real_prueba_FINAL_NUEVA_NUEVA_NUEVA_NUEVA.zarr"
results_dir = "results_3d_liver_segmented_barrier_Prueba2_t100_L_real_prueba_FINAL_NUEVA_NUEVA_NUEVA_NUEVA"

# ==========================================
# Run simulation
# ==========================================

if __name__ == "__main__":

    tf=100
    td=1

    run(
        use_gpu=True,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,

        nx=nx,
        ny=ny,
        nz=nz,

        tf=tf,
        td=td,

        sigma=0.15,
        params=params,
        mask_inflow=mask_vein,
        CI_values=(0.1, 0.0, 0.0, 0.0),
        masks=masks,
        out_name=out_name,
    )

    store = zarr.open(out_name, mode="a")
    store.attrs["tf"] = tf
    store.attrs["td"] = td

    # ==========================================
    # Save geometry fields in Zarr
    # ==========================================

    lobes_field = np.zeros((nx, ny, nz), dtype=np.float32)

    lobes_field[mask_left_lobe_2] = 2
    lobes_field[mask_right_lobe_8] = 8
    lobes_field[mask_central_lobe_4a] = 41
    lobes_field[mask_left_lobe_3] = 3
    lobes_field[mask_central_lobe_4b] = 42
    lobes_field[mask_post_lobe_7] = 7
    lobes_field[mask_right_lobe_5] = 5
    lobes_field[mask_post_lobe_6] = 6

    if "geometry" in store:
        del store["geometry"]

    geometry_grp = store.create_group("geometry")

    geometry_grp.create(
        "xi",
        data=mask_vein.astype(np.float32),
        chunks=(nx, ny, nz)
    )

    geometry_grp.create(
        "liver",
        data=mask_liver.astype(np.float32),
        chunks=(nx, ny, nz)
    )

    geometry_grp.create(
        "fibrosis",
        data=mask_fibrosis.astype(np.float32),
        chunks=(nx, ny, nz)
    )

    geometry_grp.create(
        "lobes",
        data=lobes_field.astype(np.float32),
        chunks=(nx, ny, nz)
    )
    # ==========================================
    # Graphics in Zarr
    # ==========================================

    export_zarr_to_vtk(
        zarr_file=out_name,
        output_dir=os.path.join(results_dir, "vtk"),
        geom_group="geometry",
        export_coefficients=True,
    )

    export_zarr_to_png(
        zarr_file=out_name,
        output_dir=os.path.join(results_dir, "plots"),
        mask_domain=mask_domain,
    )

    export_zarr_to_2d_snapshot(
        zarr_file=out_name,
        output_path=os.path.join(results_dir, "plots", "spatial_snapshot_step100.png"),
        step=100,
    )

    print("3D segmented-liver simulation completed successfully.")