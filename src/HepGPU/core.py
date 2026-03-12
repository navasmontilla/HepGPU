# ==========================================================
# CORE
# ==========================================================

from .constants import *
from .reactions import *
from .transport import *
from .zarr_utils import *
import time


def set_coefficients(tp, nx, ny, nz, masks=None, transport_values=(0.01, 0.8, 0.5, 0.5)):
    # Valores base
    d1_base, dTh_base, dTc_base, d3_base = transport_values

    # Crear arrays base
    d1 = tp.ones((nx, ny, nz)) * d1_base
    dTh = tp.ones((nx, ny, nz)) * dTh_base
    dTc = tp.ones((nx, ny, nz)) * dTc_base
    d3 = tp.ones((nx, ny, nz)) * d3_base

    # Aplicar máscaras específicas con sus valores (si existen)
    if masks is not None:
        for mask, values in masks:
            d1_val, dTh_val, dTc_val, d3_val = values
            mask_tp = tp.asarray(mask, dtype=bool)
            d1[mask_tp] = d1_val
            dTh[mask_tp] = dTh_val
            dTc[mask_tp] = dTc_val
            d3[mask_tp] = d3_val

    return d1, dTh, dTc, d3


def run(
    use_gpu,          # True para GPU (CuPy), False para CPU (NumPy)
    Lx, Ly, Lz,   # tamaño del dominio
    nx, ny, nz,       # número de nodos en cada dirección
    tf, td,              # tiempo final y delta de guardado
    sigma=0.15,                 # CFL para difusión
    params=None,
    mask_inflow=None,                     #mascara condicion inicial
    CI_values=(0.1, 0.0, 0.0, 0.0),
    masks=None,
    out_name = "output.zarr"
):

    # -----------------------
    # Choose computation device
    # -----------------------
    if use_gpu:
        import cupy as tp
    else:
        import numpy as tp
        
    # -----------------------
    # Updating parameters
    # -----------------------        
    if params is None:
        params = {}

    p = DEFAULT_PARAMETERS.copy()
    p.update(params)

    set_reaction_params(p)
    d1_base = p["d1"]
    dTh_base = p["dTh"]
    dTc_base = p["dTc"]
    d3_base = p["d3"]
    
    # -----------------------
    # Grid
    # -----------------------
    dx, dy, dz = Lx / (nx - 1), Ly / (ny - 1), Lz / (nz - 1)
    x = tp.linspace(0, Lx, nx)
    y = tp.linspace(0, Ly, ny)
    z = tp.linspace(0, Lz, nz)
    X, Y, Z = tp.meshgrid(x, y, z, indexing='ij')


    # Inicialización de la matriz de cada variable
    q1 = tp.zeros((nx, ny, nz))
    Th = tp.zeros((nx, ny, nz))
    Tc = tp.zeros((nx, ny, nz))
    q3 = tp.zeros((nx, ny, nz))
    xi = tp.zeros((nx, ny, nz))

    transport_values = (d1_base, dTh_base, dTc_base, d3_base)
    d1, dTh, dTc, d3 = set_coefficients(tp, nx, ny, nz, masks, transport_values=transport_values)

    # Averaging diffusion coefficient at interfaces between nodes
    d1x,d1y,d1z = avg_coefficients_3d(tp,d1)
    Thx,Thy,Thz = avg_coefficients_3d(tp,dTh)
    dTcx,dTcy,dTcz = avg_coefficients_3d(tp,dTc)
    d3x,d3y,d3z = avg_coefficients_3d(tp,d3)
    
    # Initial condition (en todo el dominio)
    q1[:], Th[:], Tc[:], q3[:] = CI_values

    if masks is not None:
        for mask, values in masks:
            d1_val, dTh_val, dTc_val, d3_val = values
            if d1_val == 0 and dTh_val == 0 and dTc_val == 0 and d3_val == 0:
                q1[mask] = 0.0
                Th[mask] = 0.0
                Tc[mask] = 0.0
                q3[mask] = 0.0
    #q1[maskCI] = 0.0 #esto es para hacer cero rl virus en inflow.
    
    # cálculo del dt basado en la condición de estabilidad (CFL) para difusión
    bmax = tp.max(tp.array([tp.max(d1), tp.max(dTh), tp.max(d3)]))
    dt = sigma * (min(dx, dy, dz)**2) / bmax  #Now the stability condition needs to check dx,dy and dz
    nt = int(tf / dt) # número de pasos de tiempo

    # Intervalo de guardado
    save_every = max(1, int(td / dt))  # número de pasos entre guardados
    print(f"Grid: {nx}x{ny}x{nz}, dt={dt:.5e}, nt={nt}, device={'GPU' if use_gpu else 'CPU'}")
    print(f"Guardando cada {save_every} pasos (≈{td} s simulados)")

    # Region inflow
    mask_inflow_tp = tp.asarray(mask_inflow, dtype=bool)
    xi[mask_inflow_tp] = 1.0
    count = tp.sum(mask_inflow_tp) 
    xi=xi/(count*dx*dy*dz+1.e-16) 

    print("Integral de xi =", tp.sum(xi) * dx * dy * dz)
    print(count)

    # -----------------------
    # Preparar almacenamiento Zarr
    # -----------------------
    
    if os.path.exists(out_name):
        os.system(f"rm -rf {out_name}")  

    store = zarr.open(out_name, mode='w')

    store, arrays, coef_arrays = create_zarr_store(out_name, nt, save_every, nx, ny, nz)

    save_initial_state(tp, use_gpu, arrays, coef_arrays,
                                  variables=(q1, Th, Tc, q3),
                                  coef_values=(d1, dTh, dTc, d3),
                                  store=store, dx=dx, dy=dy, dz=dz)

    # -----------------------
    # Time stepping (explicit Euler)
    # -----------------------
    start = time.time()
    save_index = 1

    for n in range(1, nt + 1):
        q1_new = q1.copy()  #esto se hace para no tener que generar u_new cada vez, y es necesario para no sobrescribir u.
        Th_new = Th.copy()
        Tc_new = Tc.copy()
        q3_new = q3.copy()

        # Cálculo del laplaciano
        lap_q1 = laplacian_3d(q1, d1x, d1y, d1z, dx, dy, dz)
        lap_Th = laplacian_3d(Th, Thx, Thy, Thz, dx, dy, dz)
        lap_q3 = laplacian_3d(q3, d3x, d3y, d3z, dx, dy, dz)

        # Cálculo de términos fuente/reacción
        q1_integral = tp.sum(q1) * dx * dy * dz  # integral of q1 over the domain

        # Cálculo de términos reacción, solo nodos interiores
        R1 = R_q1(q1[1:-1, 1:-1, 1:-1], Tc[1:-1, 1:-1, 1:-1])
        RTh = R_Th(q1[1:-1, 1:-1, 1:-1], Th[1:-1, 1:-1, 1:-1], xi[1:-1, 1:-1, 1:-1], q1_integral)
        RTc = R_Tc(q1[1:-1, 1:-1, 1:-1], Tc[1:-1, 1:-1, 1:-1], xi[1:-1, 1:-1, 1:-1], q1_integral)
        R3 = R_q3(q1[1:-1, 1:-1, 1:-1], Th[1:-1, 1:-1, 1:-1], q3[1:-1, 1:-1, 1:-1])

        # Cálculo del término de advección para Tc
        adv_x, adv_y, adv_z, vmax = adv_Tc(tp, dTcx, dTcy, dTcz, dx, dy, dz, Tc, q3)
        adv_total = adv_x + adv_y + adv_z

        # Condición CFL para advección
        dtAdv = sigma * min(dx, dy, dz) / vmax
        if (dtAdv < dt):
            raise RuntimeError(f"Violación CFL: dtAdv={dtAdv:.3e} < dt={dt:.3e} en paso n={n}")  
        
        q1_new[1:-1, 1:-1, 1:-1] += dt * (lap_q1 + R1) # + dt* R
        Th_new[1:-1, 1:-1, 1:-1] += dt * (lap_Th + RTh)
        Tc_new[1:-1, 1:-1, 1:-1] += dt * (-adv_total + RTc)  # restamos el termino de adveccion
        q3_new[1:-1, 1:-1, 1:-1] += dt * (lap_q3 + R3)

        # Neumann BC (zero flux). Por simplicidad, poner asi las cc:
        for u in (q1_new, Th_new, Tc_new, q3_new):
            u[0, :, :] = u[1, :, :]
            u[-1, :, :] = u[-2, :, :]
            u[:, 0, :] = u[:, 1, :]
            u[:, -1, :] = u[:, -2, :]
            u[:, :, 0] = u[:, :, 1]
            u[:, :, -1] = u[:, :, -2]

        # Actualizar variables
        q1 = q1_new
        Th = Th_new
        Tc = Tc_new
        q3 = q3_new


        if use_gpu:
            tp.cuda.Stream.null.synchronize() #esto es necesario para la GPU

        # Guardar cada td segundos simulados
        if n % save_every == 0: # or n == nt => si pongo eso me da error
            save_time_step(tp, use_gpu, arrays=arrays,
                              variables=(q1, Th, Tc, q3),
                              save_index=save_index)
            save_index += 1


    end = time.time()
    print(f"Simulation time: {end - start:.3f} seconds")
    print(f"Datos guardados en: {out_name}")