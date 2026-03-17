import os
import numpy as np
import zarr
from pyevtk.hl import imageToVTK
import matplotlib.pyplot as plt
import imageio


def create_zarr_store(out_name, nt, save_every, nx, ny, nz):
    """
    Create a Zarr store to save simulation variables and transport coefficients.

    Parameters
    ----------
    out_name : str
        Output path of the Zarr store.
    nt : int
        Total number of time steps.
    save_every : int
        Number of time steps between saved states.
    nx, ny, nz : int
        Number of grid nodes along the x, y, and z directions.

    Returns
    -------
    store : zarr.Group
        Root Zarr store.
    arrays : tuple
        Tuple containing the variable arrays `(q1, Th, Tc, q3)`.
    coef_arrays : tuple
        Tuple containing the coefficient arrays `(d1, dTh, dTc, d3)`.
    """
    if os.path.exists(out_name):
        os.system(f"rm -rf {out_name}")  

    store = zarr.open(out_name, mode='w')
    variables_grp = store.create_group("variables")
    difusion_adveccion_grp = store.create_group("coeficients")

    # State variables arrays
    q1_array = variables_grp.create("q1", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    Th_array = variables_grp.create("Th", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    Tc_array = variables_grp.create("Tc", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    q3_array = variables_grp.create("q3", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')

    # Transport coefficients arrays
    d1_array = difusion_adveccion_grp.create("d1", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    dTh_array = difusion_adveccion_grp.create("dTh", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    dTc_array = difusion_adveccion_grp.create("dTc", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    d3_array = difusion_adveccion_grp.create("d3", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')

    return store, (q1_array, Th_array, Tc_array, q3_array), (d1_array, dTh_array, dTc_array, d3_array)


def save_initial_state(tp, use_gpu, arrays, coef_arrays, variables, coef_values, store, dx, dy, dz):
    """
    Save the initial simulation state and transport coefficients to a Zarr store.

    Parameters
    ----------
    tp : module
        Numerical backend, typically `numpy` or `cupy`.
    use_gpu : bool
        If True, arrays are transferred from GPU to CPU before saving.
    arrays : tuple
        Tuple of Zarr arrays for the state variables `(q1, Th, Tc, q3)`.
    coef_arrays : tuple
        Tuple of Zarr arrays for the transport coefficients `(d1, dTh, dTc, d3)`.
    variables : tuple
        Tuple containing the current model variables `(q1, Th, Tc, q3)`.
    coef_values : tuple
        Tuple containing the coefficient fields `(d1, dTh, dTc, d3)`.
    store : zarr.Group
        Root Zarr store.
    dx, dy, dz : float
        Grid spacings along the x, y, and z directions.

    Returns
    -------
    None
    """
    q1_array, Th_array, Tc_array, q3_array = arrays
    d1_array, dTh_array, dTc_array, d3_array = coef_arrays
    q1, Th, Tc, q3 = variables
    d1, dTh, dTc, d3 = coef_values

    # Store grid spacings and device info as attributes
    store.attrs["dx"] = dx
    store.attrs["dy"] = dy
    store.attrs["dz"] = dz
    store.attrs["device"] = "GPU" if use_gpu else "CPU"

    # Save initial state and coefficients at index 0
    if use_gpu:
        q1_array[0] = tp.asnumpy(q1)
        Th_array[0] = tp.asnumpy(Th)
        Tc_array[0] = tp.asnumpy(Tc)
        q3_array[0] = tp.asnumpy(q3)

        d1_array[0] = tp.asnumpy(d1)
        dTh_array[0] = tp.asnumpy(dTh)
        dTc_array[0] = tp.asnumpy(dTc)
        d3_array[0] = tp.asnumpy(d3)
    else:
        q1_array[0] = q1
        Th_array[0] = Th
        Tc_array[0] = Tc
        q3_array[0] = q3

        d1_array[0] = d1
        dTh_array[0] = dTh
        dTc_array[0] = dTc
        d3_array[0] = d3
        
        
        
def save_time_step(tp, use_gpu, arrays, variables, save_index):
    """
    Save a simulation time step to the Zarr arrays.

    Parameters
    ----------
    tp : module
        Numerical backend, typically `numpy` or `cupy`.
    use_gpu : bool
        If True, arrays are transferred from GPU to CPU before saving.
    arrays : tuple
        Tuple of Zarr arrays `(q1_array, Th_array, Tc_array, q3_array)`.
    variables : tuple
        Tuple containing the current model variables `(q1, Th, Tc, q3)`.
    save_index : int
        Time index where the variables will be stored.

    Returns
    -------
    None
    """
    q1_array, Th_array, Tc_array, q3_array = arrays
    q1, Th, Tc, q3 = variables

    if use_gpu:
        q1_array[save_index] = tp.asnumpy(q1)
        Th_array[save_index] = tp.asnumpy(Th)
        Tc_array[save_index] = tp.asnumpy(Tc)
        q3_array[save_index] = tp.asnumpy(q3)
    else:
        q1_array[save_index] = q1
        Th_array[save_index] = Th
        Tc_array[save_index] = Tc
        q3_array[save_index] = q3
        
        
def export_zarr_to_vtk(zarr_file="output.zarr", output_dir="vtk_output", variables_group="variables", coef_group="coeficients", geom_group=None):
    """
    Export a Zarr simulation to VTK image files (.vti).

    Each saved time step is exported as a separate `.vti` file containing the
    four model variables as point data.

    Parameters
    ----------
    zarr_file : str, optional
        Path to the Zarr simulation file.
    output_dir : str, optional
        Output directory where `.vti` files will be written.
    variables_group : str, optional
        Name of the Zarr group containing the model variables.
    coef_group : str, optional
        Name of the Zarr group containing transport coefficients.
    geom_group : str or None, optional
        Optional name of a geometry group to be read in the future.

    Returns
    -------
    None
    """
    # Open the Zarr store in read mode
    store = zarr.open(zarr_file, mode="r")

    variables_grp = store[variables_group]
    difusion_adveccion_grp = store[coef_group]
    
    # Read variable arrays
    q1 = variables_grp["q1"]
    Th = variables_grp["Th"]
    Tc = variables_grp["Tc"]
    q3 = variables_grp["q3"]
    
    # Read coefficient arrays if needed
    d1 = difusion_adveccion_grp["d1"]
    dTh = difusion_adveccion_grp["dTh"]
    dTc = difusion_adveccion_grp["dTc"]
    d3 = difusion_adveccion_grp["d3"]

    # Read geometry if specified (currently ignored)
    xi = None
    #geometria_grp = store[geom_group]
    #xi = geometria_grp["xi"]

    dx = store.attrs["dx"]
    dy = store.attrs["dy"]
    dz = store.attrs["dz"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exportando {q1.shape[0]} pasos de tiempo a .vti ...")

    for i in range(q1.shape[0]):
        # Conert arrays to numpy if they are Zarr arrays
        q1_i = np.array(q1[i])
        Th_i = np.array(Th[i])
        Tc_i = np.array(Tc[i])
        q3_i = np.array(q3[i])

        output_path = os.path.join(output_dir, f"variables_{i:04d}")

        # Save the current time step as a VTK image file with the variables as point data
        imageToVTK(
            output_path,
            origin=(0.0, 0.0, 0.0),
            spacing=(dx, dy, dz),
            pointData={
                "q1": q1_i,
                "Th": Th_i,
                "Tc": Tc_i,
                "q3": q3_i
            }
        )
        print(f" Guardado: {output_path}.vti")

    print("Exportación completada.")
    
    
def export_zarr_to_png(
    zarr_file="output.zarr",
    output_dir="png_output",
    variables_group="variables",
    make_animation=True,
    fps=10
):
    """
    Export simulation stored in Zarr to PNG images and optionally create an animation.

    Each frame contains 4 subplots:
        q1, Th, Tc, q3 along x (central slice)

    Parameters
    ----------
    zarr_file : str
        Path to Zarr simulation file.

    output_dir : str
        Directory where PNG images will be saved.

    variables_group : str
        Zarr group containing variables.

    make_animation : bool
        If True, generate a GIF animation.

    fps : int
        Frames per second for the animation.
    """
    os.makedirs(output_dir, exist_ok=True)

    store = zarr.open(zarr_file, mode="r")
    variables = store[variables_group]

    q1 = variables["q1"]
    Th = variables["Th"]
    Tc = variables["Tc"]
    q3 = variables["q3"]

    dx = store.attrs["dx"]

    nt = q1.shape[0]
    nx = q1.shape[1]

    x = np.arange(nx) * dx

    # Central slice indices
    j = q1.shape[2] // 2
    k = q1.shape[3] // 2

    # initial state for comparison in plots
    q1_ini = np.array(q1[0, :, j, k])
    Th_ini = np.array(Th[0, :, j, k])
    Tc_ini = np.array(Tc[0, :, j, k])
    q3_ini = np.array(q3[0, :, j, k])

    print(f"Exporting {nt} PNG frames...")

    frame_paths = []

    for i in range(nt):

        q1_now = np.array(q1[i, :, j, k])
        Th_now = np.array(Th[i, :, j, k])
        Tc_now = np.array(Tc[i, :, j, k])
        q3_now = np.array(q3[i, :, j, k])

        fig, axs = plt.subplots(2, 2, figsize=(12, 5))

        axs[0,0].plot(x, q1_ini, '--', label="initial")
        axs[0,0].plot(x, q1_now, label=f"t={i}")
        axs[0,0].set_title("Virus (q1)")
        axs[0,0].set_ylabel("q1(x)")
        axs[0,0].legend()

        axs[0,1].plot(x, Th_ini, '--', label="initial")
        axs[0,1].plot(x, Th_now, label=f"t={i}")
        axs[0,1].set_title("Th cells")
        axs[0,1].set_ylabel("Th(x)")
        axs[0,1].legend()

        axs[1,0].plot(x, Tc_ini, '--', label="initial")
        axs[1,0].plot(x, Tc_now, label=f"t={i}")
        axs[1,0].set_title("Tc cells")
        axs[1,0].set_ylabel("Tc(x)")
        axs[1,0].legend()

        #axs[1,1].plot(x, q3_ini, '--', label="initial")
        axs[1,1].plot(x, q3_now, label=f"t={i}")
        axs[1,1].set_title("Cytokines (q3)")
        axs[1,1].set_ylabel("q3(x)")
        axs[1,1].legend()

        for ax in axs.flat:
            ax.set_xlabel("space (x)")

        plt.tight_layout()

        fname = os.path.join(output_dir, f"frame_{i:04d}.png")
        plt.savefig(fname)
        plt.close()

        frame_paths.append(fname)

        print(f"Saved {fname}")

    print("PNG export completed.")

    # Create animation
    if make_animation:

        gif_path = os.path.join(output_dir, "animation.gif")

        print("Creating animation...")

        with imageio.get_writer(gif_path, mode="I", fps=fps, loop=0) as writer:
            for frame in frame_paths:
                image = imageio.imread(frame)
                writer.append_data(image)

        print(f"Animation saved to {gif_path}")