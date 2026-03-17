# ==========================================================
# REACTION TERMS
# ==========================================================

from .constants import *

# Definition of reaction terms for the model equations. 
def f1M1(q1):
    """
    Viral self-dynamics term.

    This term models the intrinsic viral dynamics, including the carrying
    capacity and Allee-type effect.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.

    Returns
    -------
    array_like
        Contribution of the intrinsic viral dynamics.
    """
    return a1*q1*(C1-q1)*((q1-epsilon)/(q1+kappa))

def f1M5(q1,Tc):
    """
    Viral clearance mediated by cytotoxic T cells.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Tc : array_like
        Cytotoxic T-cell concentration.

    Returns
    -------
    array_like
        Viral loss term due to immune clearance.
    """
    return -a5*q1*Tc

def fthM2(q1_integral,Th,xi):
    """
    Recruitment of helper T cells through the inflow region.

    Parameters
    ----------
    q1_integral : float
        Integral of the viral concentration over the whole domain.
    Th : array_like
        Helper T-cell concentration.
    xi : array_like
        Normalized inflow mask.

    Returns
    -------
    array_like
        Recruitment term for helper T cells.
    """
    return a2h*(Cth-Th)*xi*q1_integral  

def fthM6(q1,Th):
    """
    Natural decay term of helper T cells.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Th : array_like
        Helper T-cell concentration.

    Returns
    -------
    array_like
        Decay term for helper T cells.
    """
    return -a6h*Th*(C1-q1)

def ftcM2(q1_integral,Tc,xi):
    """
    Recruitment of cytotoxic T cells through the inflow region.

    Parameters
    ----------
    q1_integral : float
        Integral of the viral concentration over the whole domain.
    Tc : array_like
        Cytotoxic T-cell concentration.
    xi : array_like
        Normalized inflow mask.

    Returns
    -------
    array_like
        Recruitment term for cytotoxic T cells.
    """
    return a2c*(Ctc-Tc)*xi*q1_integral   

def ftcM6(q1,Tc):
    """
    Natural decay term of cytotoxic T cells.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Tc : array_like
        Cytotoxic T-cell concentration.

    Returns
    -------
    array_like
        Decay term for cytotoxic T cells.
    """
    return -a6c*Tc*(C1-q1)

def f3M3(q1,Th,q3):
    """
    Cytokine production and decay term.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Th : array_like
        Helper T-cell concentration.
    q3 : array_like
        Cytokine concentration.

    Returns
    -------
    array_like
        Net cytokine reaction term.
    """
    return a3*Th*q1-a_nd*q3


# Definition of total reaction terms for each equation, combining the individual contributions.
def R_q1(q1,Tc):
    """
    Total reaction term for the virus equation.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Tc : array_like
        Cytotoxic T-cell concentration.

    Returns
    -------
    array_like
        Total reaction term for q1.
    """
    return f1M1(q1) + f1M5(q1,Tc)          # virus (q1)

def R_Th(q1,Th,xi,q1_integral):
    """
    Total reaction term for the helper T-cell equation.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Th : array_like
        Helper T-cell concentration.
    xi : array_like
        Normalized inflow mask.
    q1_integral : float
        Integral of the viral concentration over the whole domain.

    Returns
    -------
    array_like
        Total reaction term for Th.
    """
    return fthM2(q1_integral,Th,xi) + fthM6(q1,Th)     # células T helper (Th)

def R_Tc(q1,Tc,xi,q1_integral):
    """
    Total reaction term for the cytotoxic T-cell equation.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Tc : array_like
        Cytotoxic T-cell concentration.
    xi : array_like
        Normalized inflow mask.
    q1_integral : float
        Integral of the viral concentration over the whole domain.

    Returns
    -------
    array_like
        Total reaction term for Tc.
    """
    return ftcM2(q1_integral,Tc,xi) + ftcM6(q1,Tc)     # células T cytotoxic (Tc)

def R_q3(q1,Th,q3):
    """
    Total reaction term for the cytokine equation.

    Parameters
    ----------
    q1 : array_like
        Viral concentration.
    Th : array_like
        Helper T-cell concentration.
    q3 : array_like
        Cytokine concentration.

    Returns
    -------
    array_like
        Total reaction term for q3.
    """
    return f3M3(q1,Th,q3)                  # citoquinas (q3)


# Function to set reaction parameters from a dictionary, allowing for easy parameter updates without modifying the core reaction functions.
def set_reaction_params(params):
    """
    Update the global reaction parameters used by the model.

    This function assigns the values stored in `params` to the module-level
    variables used by the reaction-term functions.

    Parameters
    ----------
    params : dict
        Dictionary containing the reaction parameters.

    Returns
    -------
    None
    """
    global a1, C1, Cth, Ctc, epsilon, kappa, a5, a2h, a2c, a6h, a6c, a3, a_nd
    a1 = params["a1"]
    C1 = params["C1"]
    Cth = params["Cth"]
    Ctc = params["Ctc"]
    epsilon = params["epsilon"]
    kappa = params["kappa"]
    a5 = params["a5"]
    a2h = params["a2h"]
    a2c = params["a2c"]
    a6h = params["a6h"]
    a6c = params["a6c"]
    a3 = params["a3"]
    a_nd = params["a_nd"]