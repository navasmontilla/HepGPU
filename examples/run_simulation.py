import numpy as np
from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk, export_zarr_to_png
import sys
import os


# ==========================================
# Grid del test (only for defining masks)
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

# Definition of the cylindrical domain (taking the x-axis as input)
xc, yc, zc = 0, 0, 0 # center of the domain in Z (XY plane)
R_xy = np.sqrt((X - xc)**2 + (Y - yc)**2 +  (Z - zc)**2) # distance to the center of the domain in Z (XY plane)

R_inside = 0.21 * min(Lx, Ly, Lz)            # radius of the interior cylinder (maximum 0.5)
R_barrier_in = 0.70 * min(Lx, Ly, Lz)        # radius of the barrier cylinder (maximum 0.5)
R_barrier_out = 0.90 * min(Lx, Ly, Lz)       # exterior radius of the barrier cylinder (maximum 0.5)

mask_inflow = (R_xy <= R_inside) # True inside the cylinder
mask_barrier = ( (X >= 0.5) & (X <= 0.6) & (Y >= 0.0) & (Y <= 0.80) & (Z >= 0.0) & (Z <= 1) ) # True inside the barrier

# transport values inside the mask
barrier_values = (0.0, 0.0, 0.0, 0.0)

masks = [
    (mask_barrier, barrier_values)
]

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

        tf=1 ,    
        td=5,
        
        params={
            "a5": 0.06,
            "a3": 0.5
        },

        mask_inflow=mask_inflow,

        CI_values=(1.0, 0.0, 0.0, 0.0), 

        masks=masks,
        
        out_name = "output.zarr"
    )

    export_zarr_to_vtk(
        zarr_file="output.zarr",
        output_dir="results/vtk"
    )

    export_zarr_to_png(
        zarr_file="output.zarr",
        output_dir="results/plots"
    )

    print("Test completed successfully.")