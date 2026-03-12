# ==========================================================
# Reactions
# ==========================================================

from .constants import *


def f1M1(q1):
    return a1*q1*(C1-q1)*((q1-epsilon)/(q1+kappa))

def f1M5(q1,Tc):
    return -a5*q1*Tc

def fthM2(q1_integral,Th,xi):
    return a2h*(Cth-Th)*xi*q1_integral  

def fthM6(q1,Th):
    return -a6h*Th*(C1-q1)

def ftcM2(q1_integral,Tc,xi):
    return a2c*(Ctc-Tc)*xi*q1_integral   

def ftcM6(q1,Tc):
    return -a6c*Tc*(C1-q1)

def f3M3(q1,Th,q3):
    return a3*Th*q1-a_nd*q3

"""##### 2. Definimos ecuaciones finales de reacción"""

def R_q1(q1,Tc):
    return f1M1(q1) + f1M5(q1,Tc)          # virus (q1)

def R_Th(q1,Th,xi,q1_integral):
    return fthM2(q1_integral,Th,xi) + fthM6(q1,Th)     # células T helper (Th)

def R_Tc(q1,Tc,xi,q1_integral):
    return ftcM2(q1_integral,Tc,xi) + ftcM6(q1,Tc)     # células T cytotoxic (Tc)

def R_q3(q1,Th,q3):
    return f3M3(q1,Th,q3)                  # citoquinas (q3)


def set_reaction_params(params):
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