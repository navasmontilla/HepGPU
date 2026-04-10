import numpy as np
import os

from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png


# ==========================================
# Select case
# ==========================================

CASE = "chronic"   # "chronic" or "recovery"


# ==========================================
# Grid
# ==========================================

nx, ny, nz = 41, 3, 3
Lx = Ly = Lz = 1.0

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# ==========================================
# Define masks
# ==========================================

mask_inflow = (X <= 0.1)


# ==========================================
# Model parameters
# ==========================================

common_params = {
    "a1": 1.0,
    "C1": 1.0,
    "epsilon": 0.05,
    "kappa": 0.01,
    "a2h": 2.0,
    "Cth": 8.0,
    "a6h": 0.2,
    "a2c": 2.0,
    "Ctc": 15.0,
    "a6c": 0.2,
    "a3": 0.8,
    "a_nd": 0.6,
}

if CASE == "chronic":
    params = {
        **common_params,
        "a5": 0.05,
        "d1": 0.6,
        "dTh": 0.9,
        "dTc": 0.5,
        "d3": 0.5,
    }

elif CASE == "recovery":
    params = {
        **common_params,
        "a5": 0.08,
        "d1": 0.01,
        "dTh": 0.5,
        "dTc": 0.1,
        "d3": 0.06,
    }

else:
    raise ValueError("CASE must be 'chronic' or 'recovery'.")


out_name = f"output_1d_{CASE}.zarr"
results_dir = f"results_1d_{CASE}"


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

    print(f"1D {CASE} simulation completed successfully.")