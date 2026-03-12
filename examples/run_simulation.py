import numpy as np
from HepGPU.core import run
from HepGPU.zarr_utils import export_zarr_to_vtk
import sys
import os


# ==========================================
# Grid del test (solo para definir máscaras)
# ==========================================

nx, ny, nz = 41, 41, 41
Lx = Ly = Lz = 1.0

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# ==========================================
# Máscaras)
# ==========================================

# Definición del dominio cilíndrico (tomando como entrada el eje x)
xc, yc, zc = 0, 0, 0 # centro del dominio
# Coordenada radial y angular
R_xy = np.sqrt((X - xc)**2 + (Y - yc)**2 +  (Z - zc)**2) # distancia al centro del dominio en Z (plano XY)

R_interior = 0.21 * min(Lx, Ly, Lz)          # radio del cilindro interior (máximo 0.5)
R_barrera_in = 0.70 * min(Lx, Ly, Lz)           # radio del cilindro barrera (máximo 0.5)
R_barrera_out = 0.90 * min(Lx, Ly, Lz)         # radio exterior del cilindro barrera (máximo 0.5)

mask_inflow = (R_xy <= R_interior) # True dentro del cilindro
mask_barrera = ( (X >= 0.5) & (X <= 0.6) & (Y >= 0.0) & (Y <= 0.80) & (Z >= 0.0) & (Z <= 1) )# True dentro de la barrera


# valores transporte dentro de la máscara
barrier_values = (0.0, 0.0, 0.0, 0.0)


masks = [
    (mask_barrera, barrier_values)
]


# ==========================================
# Ejecutar simulación
# ==========================================

if __name__ == "__main__":
    run(
        use_gpu=True,

        Lx=Lx,
        Ly=Ly,
        Lz=Lz,

        nx=nx,
        ny=ny,
        nz=nz,

        tf=60 ,    
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
        output_dir="vtk_test"
    )
    
       

    print("Test completado")