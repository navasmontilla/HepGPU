import numpy as np
import os
from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png

# ==========================================
# Grid
# ==========================================

nx, ny, nz = 81, 3, 3
Lx = Ly = Lz = 1.0

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# ==========================================
# Define masks
# ==========================================

mask_inflow = (X <= 0.1)

mask_block = (X > Lx / 2.0 - 0.1) & (X < Lx / 2.0 + 0.1)

block_values = (0.0, 0.0, 0.0, 0.0)

masks = [
    (mask_block, block_values)
]

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

out_name = "output_1d_heterogeneous_domain.zarr"
results_dir = "results_1d_heterogeneous_domain"

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

    print("1D heterogeneous-domain simulation completed successfully.")