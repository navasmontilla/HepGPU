# ==========================================================
# CORE SIMULATION ROUTINES
# ========================================================== 

from .constants import *
from .reactions import *
from .transport import *
from .zarr_utils import *
import os
import time


def set_coefficients(tp, nx, ny, nz, masks=None, transport_values=(0.01, 0.8, 0.5, 0.5)):
    """
    Create spatially distributed transport coefficient fields.

    By default, uniform coefficient arrays are created over the full domain.
    If `masks` is provided, the values inside each mask are overwritten with
    the corresponding coefficients.

    Parameters
    ----------
    tp : module
        Numerical backend, typically `numpy` or `cupy`.
    nx, ny, nz : int
        Number of grid nodes along the x, y, and z directions.
    masks : list of tuple, optional
        List of `(mask, values)` pairs. Each `mask` must be a boolean array
        with shape `(nx, ny, nz)`, and `values` must be a 4-tuple
        `(d1, dTh, dTc, d3)`.
    transport_values : tuple of float, optional
        Default transport coefficients `(d1, dTh, dTc, d3)` used to initialize
        the full domain.

    Returns
    -------
    d1, dTh, dTc, d3 : tuple of array_like
        Spatially distributed transport coefficient fields.
    """
    # Unpack default transport values
    d1_base, dTh_base, dTc_base, d3_base = transport_values

    # Initialize coefficient arrays with default values
    d1 = tp.ones((nx, ny, nz)) * d1_base
    dTh = tp.ones((nx, ny, nz)) * dTh_base
    dTc = tp.ones((nx, ny, nz)) * dTc_base
    d3 = tp.ones((nx, ny, nz)) * d3_base

    # Overwrite coefficients inside user-defined masks
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
    use_gpu,                            # True for GPU (CuPy), False for CPU (NumPy)
    Lx, Ly, Lz,                         # length of the domain in x, y, z
    nx, ny, nz,                         # number of grid nodes in x, y, z
    tf, td,                             # final time, time interval between saved states
    sigma=0.15,                         # CFL for diffusion and advection
    params=None,
    mask_inflow=None,                   # mask boolean for inflow region
    CI_values=(0.1, 0.0, 0.0, 0.0),
    masks=None,
    out_name = "output.zarr"
):
    """
    Run a 3D HepGPU simulation.

    The model solves a coupled reaction-diffusion-advection system for the
    dynamics of virus, helper T cells, cytotoxic T cells, and cytokines on a
    structured Cartesian grid using an explicit Euler time-stepping scheme.

    Parameters
    ----------
    use_gpu : bool
        If True, use CuPy as the computational backend. Otherwise, use NumPy.
    Lx, Ly, Lz : float
        Physical dimensions of the computational domain.
    nx, ny, nz : int
        Number of grid nodes along the x, y, and z directions.
    tf : float
        Final simulation time.
    td : float
        Simulated time interval between saved states.
    sigma : float, optional
        Stability factor used in the diffusion and advection CFL conditions.
    params : dict, optional
        Dictionary with parameter values that overwrite `DEFAULT_PARAMETERS`.
    mask_inflow : array_like of bool, optional
        Boolean mask defining the inflow region where immune cells are recruited.
    CI_values : tuple of float, optional
        Initial values `(q1, Th, Tc, q3)` assigned over the full domain.
    masks : list of tuple, optional
        List of `(mask, values)` pairs used to define heterogeneous transport
        coefficients. Each `values` entry must be `(d1, dTh, dTc, d3)`.
    out_name : str, optional
        Output Zarr store path.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the advection CFL condition is violated during the simulation.

    Notes
    -----
    The transport coefficients are stored as 3D fields and can vary spatially.
    The simulation output is written to a Zarr store for later post-processing.
    """
    # Choose computation device
    if use_gpu:
        import cupy as tp
    else:
        import numpy as tp
        
    # Updating parameters       
    if params is None:
        params = {}

    p = DEFAULT_PARAMETERS.copy()
    p.update(params)

    set_reaction_params(p)
    d1_base = p["d1"]
    dTh_base = p["dTh"]
    dTc_base = p["dTc"]
    d3_base = p["d3"]
    
    # Cartesian grid setup
    dx, dy, dz = Lx / (nx - 1), Ly / (ny - 1), Lz / (nz - 1)
    x = tp.linspace(0, Lx, nx)
    y = tp.linspace(0, Ly, ny)
    z = tp.linspace(0, Lz, nz)
    X, Y, Z = tp.meshgrid(x, y, z, indexing='ij')


    # Initialize variables 
    q1 = tp.zeros((nx, ny, nz))
    Th = tp.zeros((nx, ny, nz))
    Tc = tp.zeros((nx, ny, nz))
    q3 = tp.zeros((nx, ny, nz))
    xi = tp.zeros((nx, ny, nz))

    # Build spatially distributed transport coefficients
    transport_values = (d1_base, dTh_base, dTc_base, d3_base)
    d1, dTh, dTc, d3 = set_coefficients(tp, nx, ny, nz, masks, transport_values=transport_values)

    # Averaging diffusion coefficient at interfaces between nodes
    d1x,d1y,d1z = avg_coefficients_3d(tp,d1)
    Thx,Thy,Thz = avg_coefficients_3d(tp,dTh)
    dTcx,dTcy,dTcz = avg_coefficients_3d(tp,dTc)
    d3x,d3y,d3z = avg_coefficients_3d(tp,d3)
    
    # Set uniform initial condition in all the domain
    q1[:], Th[:], Tc[:], q3[:] = CI_values

    # Force variables to zero inside fully blocked regions
    if masks is not None:
        for mask, values in masks:
            d1_val, dTh_val, dTc_val, d3_val = values
            if d1_val == 0 and dTh_val == 0 and dTc_val == 0 and d3_val == 0:
                q1[mask] = 0.0
                Th[mask] = 0.0
                Tc[mask] = 0.0
                q3[mask] = 0.0
    #q1[maskCI] = 0.0 
    
    # dt calculation based on the condition of stability (CFL) for diffusion
    bmax = tp.max(tp.array([tp.max(d1), tp.max(dTh), tp.max(d3)]))
    dt = sigma * (min(dx, dy, dz)**2) / bmax  #Now the stability condition needs to check dx,dy and dz
    nt = int(tf / dt) # número de pasos de tiempo

    # Save every td seconds simulated
    save_every = max(1, int(td / dt))  # number of time steps between saves
    print(f"Grid: {nx}x{ny}x{nz}, dt={dt:.5e}, nt={nt}, device={'GPU' if use_gpu else 'CPU'}")
    print(f"Guardando cada {save_every} pasos (≈{td} s simulados)")

    # Normalization of the inflow region
    mask_inflow_tp = tp.asarray(mask_inflow, dtype=bool)
    xi[mask_inflow_tp] = 1.0
    count = tp.sum(mask_inflow_tp) 
    xi=xi/(count*dx*dy*dz+1.e-16) 

    print("Integral de xi =", tp.sum(xi) * dx * dy * dz)
    print(count)

    # Prepare Zarr output store and arrays
    if os.path.exists(out_name):
        os.system(f"rm -rf {out_name}")  

    store = zarr.open(out_name, mode='w')

    store, arrays, coef_arrays = create_zarr_store(out_name, nt, save_every, nx, ny, nz)

    save_initial_state(tp, use_gpu, arrays, coef_arrays,
                                  variables=(q1, Th, Tc, q3),
                                  coef_values=(d1, dTh, dTc, d3),
                                  store=store, dx=dx, dy=dy, dz=dz)

    # Explicit Euler time integration loop
    start = time.time()
    save_index = 1

    for n in range(1, nt + 1):
        # Copy current state to new arrays for update (to avoid in-place overwriting)
        q1_new = q1.copy()  
        Th_new = Th.copy()
        Tc_new = Tc.copy()
        q3_new = q3.copy()

        # Laplacian diffusion terms (calculated at interior nodes)
        lap_q1 = laplacian_3d(q1, d1x, d1y, d1z, dx, dy, dz)
        lap_Th = laplacian_3d(Th, Thx, Thy, Thz, dx, dy, dz)
        lap_q3 = laplacian_3d(q3, d3x, d3y, d3z, dx, dy, dz)

        # Calculation of source/reaction terms
        q1_integral = tp.sum(q1) * dx * dy * dz  # integral of q1 over the domain

        # Calculation of reaction terms, only interior nodes to avoid boundary issues
        R1 = R_q1(q1[1:-1, 1:-1, 1:-1], Tc[1:-1, 1:-1, 1:-1])
        RTh = R_Th(q1[1:-1, 1:-1, 1:-1], Th[1:-1, 1:-1, 1:-1], xi[1:-1, 1:-1, 1:-1], q1_integral)
        RTc = R_Tc(q1[1:-1, 1:-1, 1:-1], Tc[1:-1, 1:-1, 1:-1], xi[1:-1, 1:-1, 1:-1], q1_integral)
        R3 = R_q3(q1[1:-1, 1:-1, 1:-1], Th[1:-1, 1:-1, 1:-1], q3[1:-1, 1:-1, 1:-1])

        # Calculation of the advection term for Tc
        adv_x, adv_y, adv_z, vmax = adv_Tc(tp, dTcx, dTcy, dTcz, dx, dy, dz, Tc, q3)
        adv_total = adv_x + adv_y + adv_z

        # CFL condition for advection
        dtAdv = sigma * min(dx, dy, dz) / vmax
        if (dtAdv < dt):
            raise RuntimeError(f"Violación CFL: dtAdv={dtAdv:.3e} < dt={dt:.3e} en paso n={n}")  
        
        # Explicit update of variables at interior nodes
        q1_new[1:-1, 1:-1, 1:-1] += dt * (lap_q1 + R1) 
        Th_new[1:-1, 1:-1, 1:-1] += dt * (lap_Th + RTh)
        Tc_new[1:-1, 1:-1, 1:-1] += dt * (-adv_total + RTc)  
        q3_new[1:-1, 1:-1, 1:-1] += dt * (lap_q3 + R3)

        # Homogeneous Neumann Boundary Conditions (zero flux)
        for u in (q1_new, Th_new, Tc_new, q3_new):
            u[0, :, :] = u[1, :, :]
            u[-1, :, :] = u[-2, :, :]
            u[:, 0, :] = u[:, 1, :]
            u[:, -1, :] = u[:, -2, :]
            u[:, :, 0] = u[:, :, 1]
            u[:, :, -1] = u[:, :, -2]

        # Update variables for the next iteration
        q1 = q1_new
        Th = Th_new
        Tc = Tc_new
        q3 = q3_new


        if use_gpu:
            tp.cuda.Stream.null.synchronize() # Necesary to ensure all GPU computations are finished before saving data to Zarr

        # Save every save_every seconds of simulation
        if n % save_every == 0:
            save_time_step(tp, use_gpu, arrays=arrays,
                              variables=(q1, Th, Tc, q3),
                              save_index=save_index)
            save_index += 1


    end = time.time()
    print(f"Simulation time: {end - start:.3f} seconds")
    print(f"Datos guardados en: {out_name}")