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
    diffusion_advection_grp = store.create_group("coefficients")

    # State variables arrays
    q1_array = variables_grp.create("q1", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    Th_array = variables_grp.create("Th", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    Tc_array = variables_grp.create("Tc", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    q3_array = variables_grp.create("q3", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')

    # Transport coefficients arrays
    d1_array = diffusion_advection_grp.create("d1", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    dTh_array = diffusion_advection_grp.create("dTh", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    dTc_array = diffusion_advection_grp.create("dTc", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')
    d3_array = diffusion_advection_grp.create("d3", shape=(nt // save_every + 1, nx, ny, nz), chunks=(1, nx, ny, nz), dtype='f4')

    return store, (q1_array, Th_array, Tc_array, q3_array), (d1_array, dTh_array, dTc_array, d3_array)


def save_initial_state(tp, use_gpu, arrays, coef_arrays, variables, coef_values, store, dx, dy, dz, td):
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
    store.attrs["td"] = td
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
        
        
def export_zarr_to_vtk(zarr_file="output.zarr", output_dir="vtk_output", variables_group="variables", coef_group="coefficients", geom_group=None):
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
    diffusion_advection_grp = store[coef_group]
    
    # Read variable arrays
    q1 = variables_grp["q1"]
    Th = variables_grp["Th"]
    Tc = variables_grp["Tc"]
    q3 = variables_grp["q3"]
    
    # Read coefficient arrays if needed
    d1 = diffusion_advection_grp["d1"]
    dTh = diffusion_advection_grp["dTh"]
    dTc = diffusion_advection_grp["dTc"]
    d3 = diffusion_advection_grp["d3"]

    # Read geometry if specified (currently ignored)
    xi = None
    #geometria_grp = store[geom_group]
    #xi = geometria_grp["xi"]

    dx = store.attrs["dx"]
    dy = store.attrs["dy"]
    dz = store.attrs["dz"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting {q1.shape[0]} time steps to .vti files...")

    for i in range(q1.shape[0]):
        # Convert arrays to numpy if they are Zarr arrays
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
        print(f"Saved: {output_path}.vti")

    print("Export completed.")
    
    
def export_zarr_to_png(
    zarr_file="output.zarr",
    output_dir="png_output",
    variables_group="variables",
    make_animation=True,
    export_time_plot=True,
    export_phase_plot=True,
    fps=10
):
    """
    Export simulation stored in Zarr to PNG images and optionally create
    an animation, temporal evolution plots, and phase diagrams.

    Outputs
    -------
    - Spatial PNG frames (central slice along x)
    - Optional GIF animation
    - Optional temporal evolution plots (domain-integrated variables)
    - Optional phase diagrams

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

    export_time_plot : bool
        If True, export temporal evolution plots.

    export_phase_plot : bool
        If True, export phase diagrams.

    fps : int
        Frames per second for the animation.

    Returns
    -------
    frame_paths : list of str
        List of paths to the generated PNG frames.
    """
    frames_dir = os.path.join(output_dir, "frames")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    store = zarr.open(zarr_file, mode="r")
    variables = store[variables_group]

    q1 = variables["q1"]
    Th = variables["Th"]
    Tc = variables["Tc"]
    q3 = variables["q3"]

    dx = store.attrs["dx"]
    dy = store.attrs["dy"] 
    dz = store.attrs["dz"] 
    td = store.attrs["td"] 

    nt = q1.shape[0]
    nx = q1.shape[1]

    x = np.arange(nx) * dx
    t = np.arange(nt) * td  # time vector based on the number of snapshots and the save interval

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
        axs[0,1].set_title("T helper (Th)")
        axs[0,1].set_ylabel("Th(x)")
        axs[0,1].legend()

        axs[1,0].plot(x, Tc_ini, '--', label="initial")
        axs[1,0].plot(x, Tc_now, label=f"t={i}")
        axs[1,0].set_title("T cytotoxic (Tc)")
        axs[1,0].set_ylabel("Tc(x)")
        axs[1,0].legend()

        axs[1,1].plot(x, q3_ini, '--', label="initial")
        axs[1,1].plot(x, q3_now, label=f"t={i}")
        axs[1,1].set_title("Cytokines (q3)")
        axs[1,1].set_ylabel("q3(x)")
        axs[1,1].legend()

        for ax in axs.flat:
            ax.set_xlabel("space (x)")

        plt.tight_layout()

        fname = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
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

    if export_time_plot or export_phase_plot:
        q1total = np.array([np.sum(np.array(q1[i])) * dx * dy * dz for i in range(nt)])
        Thtotal = np.array([np.sum(np.array(Th[i])) * dx * dy * dz for i in range(nt)])
        Tctotal = np.array([np.sum(np.array(Tc[i])) * dx * dy * dz for i in range(nt)])
        q3total = np.array([np.sum(np.array(q3[i])) * dx * dy * dz for i in range(nt)])

    # Create temporal graphs
    if export_time_plot:

        fig, axs = plt.subplots(2,2,figsize=(8, 3), constrained_layout=True)

        ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

        ax1.plot(t, q1total, 'b', label='q1(t)')
        ax2.plot(t, Thtotal, 'g', label='Th(t)')
        ax3.plot(t, Tctotal, 'y', label='Tc(t)')
        ax4.plot(t, q3total, 'r', label='q3(t)')

        ax1.set_title("Virus (q1)")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("q1(t)")

        ax2.set_title("T helper (Th)")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("Th(t)")

        ax3.set_title("T cytotoxic (Tc)")
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("Tc(t)")

        ax4.set_title("Cytokines (q3)")
        ax4.set_xlabel("time (s)")
        ax4.set_ylabel("q3(t)")

        time_plot_path = os.path.join(output_dir, "temporal_evolution.png")
        fig.savefig(time_plot_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"Saved {time_plot_path}")

    if export_phase_plot:

        fig, axs = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)

        axs[0].plot(q1total, Tctotal, 'k')
        axs[0].plot(q1total[0], Tctotal[0], 'or', label="Start")
        axs[0].plot(q1total[-1], Tctotal[-1], 'ob', label="End")
        axs[0].set_xlabel("Virus (q1)")
        axs[0].set_ylabel("T cytotoxic (Tc)")
        axs[0].legend()

        axs[1].plot(q1total, Thtotal, 'k')
        axs[1].plot(q1total[0], Thtotal[0], 'or', label="Start")
        axs[1].plot(q1total[-1], Thtotal[-1], 'ob', label="End")
        axs[1].set_xlabel("Virus (q1)")
        axs[1].set_ylabel("T helper (Th)")
        axs[1].legend()

        axs[2].plot(q1total, q3total, 'k')
        axs[2].plot(q1total[0], q3total[0], 'or', label="Start")
        axs[2].plot(q1total[-1], q3total[-1], 'ob', label="End")
        axs[2].set_xlabel("Virus (q1)")
        axs[2].set_ylabel("Cytokines (q3)")
        axs[2].legend()

        phase_plot_path = os.path.join(output_dir, "phase_diagram.png")
        fig.savefig(phase_plot_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"Saved {phase_plot_path}")
    
    return frame_paths

def export_zarr_to_2d_snapshot(
    zarr_file="output.zarr",
    output_path="spatial_snapshot.png",
    variables_group="variables",
    step=0,
    cmap="viridis",
    dpi=300
):
    """
    Export a single 2D spatial snapshot from a Zarr simulation.

    This function is intended for effective 2D simulations stored in a thin
    3D grid (typically nz = 3). It extracts the central z-slice and creates
    one 2x2 figure with:
        - q1
        - Th
        - Tc
        - q3

    Parameters
    ----------
    zarr_file : str
        Path to the Zarr simulation file.
    output_path : str
        Output PNG file path.
    variables_group : str
        Name of the Zarr group containing the variables.
    step : int
        Snapshot index to export.
    cmap : str
        Matplotlib colormap for the contour plots.
    dpi : int
        Resolution of the saved figure.

    Returns
    -------
    None
    """
    store = zarr.open(zarr_file, mode="r")
    variables = store[variables_group]

    q1 = variables["q1"]
    Th = variables["Th"]
    Tc = variables["Tc"]
    q3 = variables["q3"]

    dx = store.attrs["dx"]
    dy = store.attrs["dy"]

    nt, nx, ny, nz = q1.shape

    if not (0 <= step < nt):
        raise ValueError(f"step must be between 0 and {nt-1}, got {step}")

    mid = nz // 2

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    data_q1 = np.array(q1[step, :, :, mid])
    cp1 = axs[0, 0].contourf(X, Y, data_q1, levels=50, cmap=cmap)
    axs[0, 0].set_title(f"Virus (q1), step {step}")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    fig.colorbar(cp1, ax=axs[0, 0], label="q1")

    data_Th = np.array(Th[step, :, :, mid])
    cp2 = axs[0, 1].contourf(X, Y, data_Th, levels=50, cmap=cmap)
    axs[0, 1].set_title(f"T helper (Th), step {step}")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    fig.colorbar(cp2, ax=axs[0, 1], label="Th")

    data_Tc = np.array(Tc[step, :, :, mid])
    cp3 = axs[1, 0].contourf(X, Y, data_Tc, levels=50, cmap=cmap)
    axs[1, 0].set_title(f"T cytotoxic (Tc), step {step}")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    fig.colorbar(cp3, ax=axs[1, 0], label="Tc")

    data_q3 = np.array(q3[step, :, :, mid])
    cp4 = axs[1, 1].contourf(X, Y, data_q3, levels=50, cmap=cmap)
    axs[1, 1].set_title(f"Cytokines (q3), step {step}")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("y")
    fig.colorbar(cp4, ax=axs[1, 1], label="q3")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05
    )
    plt.close(fig)