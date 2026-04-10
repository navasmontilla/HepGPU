import numpy as np
import os

from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png, export_zarr_to_2d_snapshot


# ==========================================
# Grid
# ==========================================

nx, ny, nz = 41, 41, 41
Lx = Ly = Lz = 1.0

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# ==========================================
# Define masks
# ==========================================

xc, yc, zc = 0.0, 0.0, 0.0
R_xyz = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)

R_inflow = 0.11 * min(Lx, Ly, Lz)
mask_inflow = (R_xyz <= R_inflow)


# ==========================================
# Model parameters
# ==========================================

params = {
    "a1": 1.0,
    "C1": 1.0,
    "epsilon": 0.05,
    "kappa": 0.01,
    "a5": 0.055,
    "a2h": 2.0,
    "Cth": 8.0,
    "a6h": 0.2,
    "a2c": 2.0,
    "Ctc": 15.0,
    "a6c": 0.2,
    "a3": 0.8,
    "a_nd": 0.6,
    "d1": 0.6,
    "dTh": 0.9,
    "dTc": 0.5,
    "d3": 0.5,
}

out_name = "output_3d_spherical_inflow.zarr"
results_dir = "results_3d_spherical_inflow"


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

        tf=50,
        td=1,
        sigma=0.15,
        params=params,
        mask_inflow=mask_inflow,
        CI_values=(0.1, 0.0, 0.0, 0.0),
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
        output_path=os.path.join(results_dir, "spatial_snapshot_step_50.png"),
        step=50,
    )

    print("3D spherical-inflow simulation completed successfully.")