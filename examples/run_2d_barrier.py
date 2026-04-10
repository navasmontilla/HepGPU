import numpy as np
import os

from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png, export_zarr_to_2d_snapshot


# ==========================================
# Grid
# ==========================================

nx, ny, nz = 101, 101, 3
Lx, Ly, Lz = 1.0, 1.0, 1.0

x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
z = np.linspace(0.0, Lz, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# ==========================================
# Define masks
# ==========================================

xc, yc = 0.0, 0.0

R_xy = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
theta = np.arctan2(Y - yc, X - xc)

R_inflow = 0.21 * min(Lx, Ly, Lz)
R_barrier_in = 0.70 * min(Lx, Ly, Lz)
R_barrier_out = 0.90 * min(Lx, Ly, Lz)

mask_inflow = R_xy <= R_inflow
mask_barrier_ring = (R_xy >= R_barrier_in) & (R_xy <= R_barrier_out)

theta0 = 0.0
w = 0.12 * min(Lx, Ly)
Rmid = 0.5 * (R_barrier_in + R_barrier_out)
dtheta = w / Rmid

mask_open_sector = np.abs(theta - theta0) < (dtheta / 2.0)
mask_barrier = mask_barrier_ring & (~mask_open_sector)

barrier_values = (0.0, 0.0, 0.0, 0.0)

masks = [
    (mask_barrier, barrier_values)
]


# ==========================================
# Model parameters
# ==========================================

params = {
    "a5": 0.06,
    "a3": 0.5,
    "dTc": 0.1,
    "d1": 0.01,
    "dTh": 0.5,
    "d3": 0.06,
}

out_name = "output_2d_barrier.zarr"
results_dir = "results_2d_barrier"


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

        tf=400.0,
        td=1.0,
        
        sigma=0.15,
        params=params,
        mask_inflow=mask_inflow,
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
        step=400,
    )

    print("2D barrier simulation completed successfully.")