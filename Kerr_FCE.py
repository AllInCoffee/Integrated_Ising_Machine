# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:31:19 2025

@author: yanxi
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
#import cmath
import math
import time
from sympy import *
from sympy import symbols
from scipy import optimize

Delta=400
u=np.arange(0,60,6) ## before 50 there are not yet bi-states
u_new=list(reversed(u))
print(u_new)
def CMT2(t,Y,Delta,u,chi,tau,k_T,gamma_TPA,gamma_FCA,sigma_FCD):
    z = Y[0]
    n = Y[1]
    dzdt = -(k_T/2+1j*(Delta+chi*abs(z)**2))*z-math.sqrt(k)*u+1j*(n+sigma_FCD*n**0.8)*z - (gamma_TPA*abs(z)**2+gamma_FCA*n)*z
    dndt = abs(z)**4-n/tau
    return [dzdt, dndt]

chi = 0.55
tau=18.5
k_l=50 * abs(chi)
k = 150 * abs(chi)
k_T=k+k_l
gamma_TPA=0.11
gamma_FCA=0.2
sigma_FCD=7.2

lower_limit = 0.0001 # not zero to avoid nan
upper_limit = 3
num_pts = 100
E_list = np.linspace(lower_limit, upper_limit, num_pts)
Delta_list=np.linspace(0,510,50)
#print(Delta_list)
u_list=[]
E_diff1=[]
E=symbols('E')
Delta=symbols('Delta') 
eqn=((k_T/2 + gamma_TPA*E + gamma_FCA * (18.5*E**2 ))**2 + (Delta + chi*E-(18.5*E**2 )-sigma_FCD*(18.5*E**2 )**0.8)**2)*E/k

# for i in Delta_list:
#     u_list=[]
#     E_diff1=[]
#     for j in E_list:
#        P=eqn.subs([(E,j),(Delta,i)])
#        u_list.append(math.sqrt(P))
#        new=eqn.diff(E).subs([(E,j),(Delta,i)])
#        E_diff1.append(new)
#     #plt.plot(u_list,E_list)
#     zero_crossings = np.where(np.diff(np.sign(E_diff1)))[0]+1
#     print(zero_crossings)
#     if len(zero_crossings)==2:
#         plt.plot([i]*2,[u_list[i] for i in zero_crossings] )
# plt.title('Bistability Diagram with Kerr and FCE')
# plt.xlabel('$\Delta$')
# plt.ylabel('u') 

#### plot bistability range 
# bi_u=[]
# bi_delta=[]
# for i in Delta_list:
#     u_list=[]
#     E_diff1=[]
#     for j in E_list:
#        P=eqn.subs([(E,j),(Delta,i)])
#        u_list.append(math.sqrt(P))
#        new=eqn.diff(E).subs([(E,j),(Delta,i)])
#        E_diff1.append(new)
#     zero_crossings = np.where(np.diff(np.sign(E_diff1)))[0]+1
#     #print(zero_crossings)
#     if len(zero_crossings)==2:
#         bi_delta.append(i)
#         bi_u.append([u_list[i] for i in zero_crossings])
# #print(len(bi_u),len(bi_u[0]))
# plt.plot(bi_delta,[x[0] for x in bi_u],'g') # append all smallest u
# plt.plot(bi_delta,[x[1] for x in bi_u],'g') # append all biggest u
# plt.title('Bistability Diagram with Kerr and FCE')
# plt.xlabel('$\Delta$')
# plt.ylabel('u')