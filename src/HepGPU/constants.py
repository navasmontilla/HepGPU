# ==========================================================
# CONSTANTES
# ==========================================================

DEFAULT_PARAMETERS = {
    # Virus dynamics
    "a1": 1.0,        # natural decay rate of the virus (q1)
    "C1": 1.0,        # carrying capacity of the virus (q1)

    # Immune cell carrying capacities
    "Cth": 8.0,       # carrying capacity of helper T cells (Th)
    "Ctc": 15.0,      # carrying capacity of cytotoxic T cells (Tc)

    # Allee effect parameters
    "epsilon": 0.05,  # Allee effect parameter
    "kappa": 0.01,    # Allee effect parameter

    # Immune response parameters
    "a5": 0.06,       # effectiveness of T cells in clearing virus (original 0.055)
                     # higher values (e.g., 0.1) may reduce sharp growth of q1

    "a2h": 2.0,      # recruitment rate of Th cells through the portal field
                     # depends on the total virus (q1) at time t

    "a2c": 2.0,      # recruitment rate of Tc cells

    # Natural decay of immune cells
    "a6h": 0.2,      # decay rate of Th cells
    "a6c": 0.2,      # decay rate of Tc cells

    # Cytokine dynamics
    "a3": 0.8,        # cytokine (q3) production by virus and Th cells
    "a_nd": 0.6,       # cytokine natural decay constant
    
        # -----------------------
    # Transport coefficients
    # -----------------------
    "d1": 0.6,   # virus diffusion
    "dTh": 0.9,   # Th diffusion
    "dTc": 4.0,   # Tc advection
    "d3": 0.5     # cytokine diffusion 
    
}

