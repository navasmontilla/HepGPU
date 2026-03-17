# ==========================================================
# CONSTANTS DEFAULT PARAMETERS
# ==========================================================

DEFAULT_PARAMETERS = {
    # Virus dynamics
    "a1": 1.0,        # Natural decay rate of the virus (q1)
    "C1": 1.0,        # Carrying capacity of the virus (q1)

    # Immune cell carrying capacities
    "Cth": 8.0,       # Carrying capacity of helper T cells (Th)
    "Ctc": 15.0,      # Carrying capacity of cytotoxic T cells (Tc)

    # Allee effect parameters
    "epsilon": 0.05,  # Allee effect parameter
    "kappa": 0.01,    # Allee effect parameter

    # Immune response parameters
    "a5": 0.06,       # Efficiency of Tc cells in clearing virus (original 0.055)
                      # Higher values (e.g., 0.1) may reduce sharp growth of q1

    "a2h": 2.0,       # Recruitment rate of Th cells through the portal field
                      # Depends on the total virus load (q1) at time t

    "a2c": 2.0,       # Recruitment rate of Tc cells through the portal field
                      # Depends on the total virus load (q1) at time t

    # Natural decay of immune cells
    "a6h": 0.2,       # Natural decay rate of Th cells
    "a6c": 0.2,       # Natural decay rate of Tc cells

    # Cytokine dynamics
    "a3": 0.8,         # Cytokine (q3) production rate driven by virus and Th cells
    "a_nd": 0.6,       # Natural decay rate of cytokine 
    
    # Transport coefficients
    "d1": 0.6,         # virus diffusion
    "dTh": 0.9,        # Th diffusion
    "dTc": 4.0,        # Tc advection
    "d3": 0.5          # Cytokine diffusion 
    
}

