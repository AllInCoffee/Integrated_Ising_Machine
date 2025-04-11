# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 07:56:40 2025

@author: yanxi
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
from itertools import cycle
from scipy import constants

# constants under 1550nm
#Typical FCA Coefficient Value for Silicon (at 1550 nm): FCA cross section
sigma=1.45e-21 # m^2 
#The heat capacity per unit volume of silicon:
C= 1.63e6 # J/m^3
#Photon energy at 1550nm:
energy_ph=0.8*constants.elementary_charge #eV

# Kerr coefficient / SPM coefficient in terms of the nonlinear phase shift,
#gamma= 2*math.pi*n2/wavelength
gamma=3.1e-11 # m/W
# TPA-to-SPM strength ratio: r = beta/(2*gamma)
r=0.19
u = 30

###################### changeable #######################
# cavity constants
#Mode effective area 
A_eff=1e-13 # m^2 
T0=0 # initial temperature 
# Photon lifetime
tau_ph = 455e-12  # 455ps 
V_rb=0.5 # V_rb >=0

# free carrier lifetime depends on photon lifetime AND V_rb
# FC-lifetime decay constant
V_fc=2.15 # V
tau_c_sat=3.1e-12 # 3.1ps FC lifetime saturation
tau_c_0=140e-12 # 140ps FC lifetime when V_rb=0
tau_c= (tau_c_0-tau_c_sat)*math.exp(-V_rb/V_fc) + tau_c_sat
print(f'\nVoltage:{V_rb} V \nfree carrier life time: {tau_c*1e12} ps')
tau_c_relative= tau_c/tau_ph
# Thermal detuning coefficient 
delta_T= -9.7e9 # s^-1*K^-1
######################################################
# fixed TO decay time: 30ns
tau_th_absolute=30e-6 
tau_th_relative= tau_th_absolute/ tau_ph # near 66*tau_ph 
# group index
n_g=3.97
v_g= constants.c / n_g
#print(v_g)
# Ring-bus power coupling
theta=1.6e-3
# Round-trip time
t_R= 1.7e-12 # 1.7 ps
## input normalied factor
scale_in= t_R / math.sqrt(8*tau_ph**3*gamma*v_g*theta)

 
# how strongly the free carriers affect the resonance frequency shift (via FCD) * how long the free carriers stay in the system
#  In summary, accumulated phase shift per photon before recombination due to the of FCD (normalized to the photon lifetim)
chi= (r*u*sigma) / (4*energy_ph*gamma*v_g) * tau_c_relative
#print(chi)


# Thermal loading coefficient (how efficiently optical power generates heat)
eta = 2*delta_T*r / (gamma*v_g**2*C)  # Vmode/Vth ==1
#print(eta)

V_fact=1+V_rb*constants.elementary_charge/(2*energy_ph)  # the second term is additional carrier heating term (related to energy per carrier)
print(f'Heating term: {V_fact}')

# Sweep settings
lower_limit = 0.0001
upper_limit = 0.2
num_pts = 50
E_list = np.linspace(lower_limit, upper_limit, num_pts)
Delta_range = np.arange(-10, 11, 5)

# Color map for different Delta values
# cmap = plt.get_cmap("tab10")
# colors = cmap(np.linspace(0, 1, len(Delta_range)))

# Get the color from the current default color cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


#plt.figure(figsize=(10, 6))

for idx, j in enumerate(Delta_range):
    a_in = []
    color=next(color_cycle)
    for i in E_list:
        T_stable=eta*tau_th_relative*(V_fact*i**2+ chi * i**3/(u*r))+T0
        P = ((1 + r * i + (1 / u) * chi * i**2)**2 + (j-T_stable - i + chi * i**2)**2) * i
        a_in.append(math.sqrt(P))
        #print('The input power is:  {:.3f}W'.format(P * scale_in**2* A_eff))
    
    a_in = np.array(a_in)
    #color = colors[idx]

    # Detect turning points
    turning_indices_max = argrelextrema(a_in, np.greater)[0]
    turning_indices_min = argrelextrema(a_in, np.less)[0]
    turning_indices = np.sort(np.concatenate((turning_indices_max, turning_indices_min)))

    # Plot with color-coding and branch styles
    if len(turning_indices) >= 2:
        plt.plot(a_in[:turning_indices[0]], E_list[:turning_indices[0]], color=color, label=f'Δ={j} (lower)')
        plt.plot(a_in[turning_indices[0]:turning_indices[1]], E_list[turning_indices[0]:turning_indices[1]], color='k',linestyle='--', label=f'Δ={j} (bistable)')
        plt.plot(a_in[turning_indices[1]:], E_list[turning_indices[1]:], color=color, label=f'Δ={j} (upper)')
        plt.plot(a_in[turning_indices], E_list[turning_indices], 'ko', markersize=5)
    else:
        plt.plot(a_in, E_list,color=color, label=f'Δ={j}')

plt.title('Bifurcation Diagram with Bistability and Color-coded Detuning')
plt.xlabel(r'$a_{\mathrm{in}}$ (Input Amplitude)')
plt.ylabel(r'$E$ (Circulating Power)')
plt.grid(True)
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()
