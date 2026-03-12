# ==========================================================
# MAIN
# ==========================================================

# Diffusion operator
def avg_coefficients_3d(tp,b): #esto calcula los valores promedio de los coef de difusion necesarios para laplacian_3d
    bx = tp.where((b[1:, 1:-1, 1:-1]  > 0) &(b[:-1, 1:-1, 1:-1] > 0),0.5 * (b[1:, 1:-1, 1:-1]  + b[:-1, 1:-1, 1:-1]), 0.0)
    by = tp.where((b[1:-1, 1:, 1:-1]  > 0) &(b[1:-1, :-1, 1:-1] > 0),0.5 * (b[1:-1, 1:, 1:-1]  + b[1:-1, :-1, 1:-1]), 0.0)
    bz = tp.where((b[1:-1, 1:-1, 1:]  > 0) &(b[1:-1, 1:-1, :-1] > 0),0.5 * (b[1:-1, 1:-1, 1:]  + b[1:-1, 1:-1, :-1]), 0.0)
    return bx, by, bz

def laplacian_3d(u, bx, by, bz, dx, dy, dz): #usaremos un laplacian3D unico para todas eqs
    return (
        (bx[1:,:,:]*(u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) - bx[:-1,:,:]*( u[1:-1, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) ) / dx**2 +
        (by[:,1:,:]*(u[1:-1, 2:, 1:-1] - u[1:-1, 1:-1, 1:-1]) - by[:,:-1,:]*( u[1:-1, 1:-1, 1:-1] - u[1:-1, :-2, 1:-1]) ) / dy**2 +
        (bz[:,:,1:]*(u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 1:-1]) - bz[:,:,:-1]*( u[1:-1, 1:-1, 1:-1] - u[1:-1, 1:-1, :-2]) ) / dz**2
    )

# Advection operator
def adv_Tc(tp,dTcx, dTcy, dTcz, dx, dy, dz, Tc, q3):    # termino chemotaxis adveccion

    # Velocidades en las caras de las celdas
    vpx=dTcx[1:,:,:]*(q3[2:, 1:-1, 1:-1]-q3[1:-1, 1:-1, 1:-1])/dx
    vmx=dTcx[:-1,:,:]*(q3[1:-1, 1:-1, 1:-1]-q3[:-2, 1:-1, 1:-1])/dx
    vpy=dTcy[:,1:,:]*(q3[1:-1, 2:, 1:-1]-q3[1:-1, 1:-1, 1:-1])/dy
    vmy=dTcy[:,:-1,:]*(q3[1:-1, 1:-1, 1:-1]-q3[1:-1, :-2, 1:-1])/dy
    vpz=dTcz[:,:,1:]*(q3[1:-1, 1:-1, 2:]-q3[1:-1, 1:-1, 1:-1])/dz
    vmz=dTcz[:,:,:-1]*(q3[1:-1, 1:-1, 1:-1]-q3[1:-1, 1:-1, :-2])/dz

    # Flujo Upwind
    fpx=tp.maximum(vpx,0)*Tc[1:-1, 1:-1, 1:-1] + tp.minimum(vpx,0)*Tc[2:, 1:-1, 1:-1]
    fmx=tp.maximum(vmx,0)*Tc[:-2, 1:-1, 1:-1] + tp.minimum(vmx,0)*Tc[1:-1, 1:-1, 1:-1]
    fpy=tp.maximum(vpy,0)*Tc[1:-1, 1:-1, 1:-1] + tp.minimum(vpy,0)*Tc[1:-1, 2:, 1:-1]
    fmy=tp.maximum(vmy,0)*Tc[1:-1, :-2, 1:-1] + tp.minimum(vmy,0)*Tc[1:-1, 1:-1, 1:-1]
    fpz=tp.maximum(vpz,0)*Tc[1:-1, 1:-1, 1:-1] + tp.minimum(vpz,0)*Tc[1:-1, 1:-1, 2:]
    fmz=tp.maximum(vmz,0)*Tc[1:-1, 1:-1, :-2] + tp.minimum(vmz,0)*Tc[1:-1, 1:-1, 1:-1]

    # vmax 
    vmax = tp.abs(vpx).max()
    vmax = tp.maximum(vmax, tp.abs(vmx).max())
    vmax = tp.maximum(vmax, tp.abs(vpy).max())
    vmax = tp.maximum(vmax, tp.abs(vmy).max())
    vmax = tp.maximum(vmax, tp.abs(vpz).max())
    vmax = tp.maximum(vmax, tp.abs(vmz).max())

    vmax = float(vmax)
    vmax = max(vmax, 1e-14)
    
    return ( fpx-fmx )/dx, ( fpy-fmy )/dy, ( fpz-fmz )/dz, vmax 