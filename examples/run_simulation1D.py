import numpy as np
from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png
import sys
import os


# ==========================================
# Grid del test (only for defining masks)
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

mask_inflow = (X <= 0.1) # True inside the cylinder

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

        tf=50 ,
        td=1,

        params={
            "a1": 1.0,       # virus natural decay
            "C1": 1.0,       # virus carrying capacity
            "epsilon": 0.05, # Allee effect parameter
            "kappa": 0.01,   # Allee effect parameter

            "a5": 0.055,     # T cell effectiveness

            "a2h": 2.0,      # Th inflow regulation
            "Cth": 8.0,      # Th carrying capacity
            "a6h": 0.2,      # Th decay rate

            "a2c": 2.0,      # Tc inflow regulation
            "Ctc": 15.0,     # Tc carrying capacity
            "a6c": 0.2,      # Tc decay rate

            "a3": 0.8,       # cytokine production
            "a_nd": 0.6,     # cytokine degradation

            "dTc": 0.5,      # Tc diffusion coefficient
            "d1": 0.6,       # virus diffusion coefficient
            "dTh": 0.9,      # Th diffusion coefficient
            "d3": 0.5        # cytokine diffusion coefficient
        },

        mask_inflow=mask_inflow,

        CI_values=(0.1, 0.0, 0.0, 0.0),

        out_name = "output1D.zarr"
    )

    export_zarr_to_vtk(
        zarr_file="output1D.zarr",
        output_dir="results1D/vtk"
    )

    export_zarr_to_png(
        zarr_file="output1D.zarr",
        output_dir="results1D/plots"
    )

    print("Test completed successfully.")