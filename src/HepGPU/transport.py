# ==========================================================
# TRANSPORT OPERATORS
# ==========================================================

# Diffusion operator
def avg_coefficients_3d(tp,b): 
    """
    Compute face-centered averaged transport coefficients in 3D.

    The function averages cell-centered coefficients at the interfaces
    between neighboring grid points. If one of the adjacent coefficients
    is non-positive, the interface coefficient is set to zero.

    Parameters
    ----------
    tp : module
        Numerical backend, typically `numpy` or `cupy`.
    b : array_like
        Cell-centered transport coefficient field.

    Returns
    -------
    bx, by, bz : tuple of array_like
        Averaged coefficients at cell faces along the x, y, and z directions.
    """
    bx = tp.where((b[1:, 1:-1, 1:-1]  > 0) &(b[:-1, 1:-1, 1:-1] > 0),0.5 * (b[1:, 1:-1, 1:-1]  + b[:-1, 1:-1, 1:-1]), 0.0)
    by = tp.where((b[1:-1, 1:, 1:-1]  > 0) &(b[1:-1, :-1, 1:-1] > 0),0.5 * (b[1:-1, 1:, 1:-1]  + b[1:-1, :-1, 1:-1]), 0.0)
    bz = tp.where((b[1:-1, 1:-1, 1:]  > 0) &(b[1:-1, 1:-1, :-1] > 0),0.5 * (b[1:-1, 1:-1, 1:]  + b[1:-1, 1:-1, :-1]), 0.0)
    return bx, by, bz

def laplacian_3d(u, bx, by, bz, dx, dy, dz): 
    """
    Compute the discrete 3D diffusion operator with variable coefficients.

    Parameters
    ----------
    u : array_like
        Scalar field defined on the grid.
    bx, by, bz : array_like
        Face-centered transport coefficients along x, y, and z.
    dx, dy, dz : float
        Grid spacings along each direction.

    Returns
    -------
    array_like
        Discrete diffusion operator evaluated on the interior nodes.
    """
    return (
        (bx[1:,:,:]*(u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) - bx[:-1,:,:]*( u[1:-1, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) ) / dx**2 +
        (by[:,1:,:]*(u[1:-1, 2:, 1:-1] - u[1:-1, 1:-1, 1:-1]) - by[:,:-1,:]*( u[1:-1, 1:-1, 1:-1] - u[1:-1, :-2, 1:-1]) ) / dy**2 +
        (bz[:,:,1:]*(u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 1:-1]) - bz[:,:,:-1]*( u[1:-1, 1:-1, 1:-1] - u[1:-1, 1:-1, :-2]) ) / dz**2
    )

# Advection operator
def adv_Tc(tp,dTcx, dTcy, dTcz, dx, dy, dz, Tc, q3):   
    """
    Compute the chemotactic advection term for cytotoxic T cells using an
    upwind flux discretization.

    The chemotactic velocity is driven by the gradient of the cytokine field
    `q3`, scaled by the transport coefficients `dTcx`, `dTcy`, and `dTcz`.

    Parameters
    ----------
    tp : module
        Numerical backend, typically `numpy` or `cupy`.
    dTcx, dTcy, dTcz : array_like
        Face-centered transport coefficients along x, y, and z.
    dx, dy, dz : float
        Grid spacings along each direction.
    Tc : array_like
        Cytotoxic T-cell concentration.
    q3 : array_like
        Cytokine concentration.

    Returns
    -------
    adv_x, adv_y, adv_z : array_like
        Discrete advection contributions along x, y, and z, evaluated on the
        interior nodes.
    vmax : float
        Maximum absolute velocity magnitude used for the advection CFL check.
    """
    # Face-centered chemotactic velocities
    vpx=dTcx[1:,:,:]*(q3[2:, 1:-1, 1:-1]-q3[1:-1, 1:-1, 1:-1])/dx
    vmx=dTcx[:-1,:,:]*(q3[1:-1, 1:-1, 1:-1]-q3[:-2, 1:-1, 1:-1])/dx
    vpy=dTcy[:,1:,:]*(q3[1:-1, 2:, 1:-1]-q3[1:-1, 1:-1, 1:-1])/dy
    vmy=dTcy[:,:-1,:]*(q3[1:-1, 1:-1, 1:-1]-q3[1:-1, :-2, 1:-1])/dy
    vpz=dTcz[:,:,1:]*(q3[1:-1, 1:-1, 2:]-q3[1:-1, 1:-1, 1:-1])/dz
    vmz=dTcz[:,:,:-1]*(q3[1:-1, 1:-1, 1:-1]-q3[1:-1, 1:-1, :-2])/dz

    # Upwind fluxes
    fpx=tp.maximum(vpx,0)*Tc[1:-1, 1:-1, 1:-1] + tp.minimum(vpx,0)*Tc[2:, 1:-1, 1:-1]
    fmx=tp.maximum(vmx,0)*Tc[:-2, 1:-1, 1:-1] + tp.minimum(vmx,0)*Tc[1:-1, 1:-1, 1:-1]
    fpy=tp.maximum(vpy,0)*Tc[1:-1, 1:-1, 1:-1] + tp.minimum(vpy,0)*Tc[1:-1, 2:, 1:-1]
    fmy=tp.maximum(vmy,0)*Tc[1:-1, :-2, 1:-1] + tp.minimum(vmy,0)*Tc[1:-1, 1:-1, 1:-1]
    fpz=tp.maximum(vpz,0)*Tc[1:-1, 1:-1, 1:-1] + tp.minimum(vpz,0)*Tc[1:-1, 1:-1, 2:]
    fmz=tp.maximum(vmz,0)*Tc[1:-1, 1:-1, :-2] + tp.minimum(vmz,0)*Tc[1:-1, 1:-1, 1:-1]

    # Maximum velocity magnitude for CFL control
    vmax = tp.abs(vpx).max()
    vmax = tp.maximum(vmax, tp.abs(vmx).max())
    vmax = tp.maximum(vmax, tp.abs(vpy).max())
    vmax = tp.maximum(vmax, tp.abs(vmy).max())
    vmax = tp.maximum(vmax, tp.abs(vpz).max())
    vmax = tp.maximum(vmax, tp.abs(vmz).max())

    vmax = float(vmax)
    vmax = max(vmax, 1e-14)
    
    return ( fpx-fmx )/dx, ( fpy-fmy )/dy, ( fpz-fmz )/dz, vmax 