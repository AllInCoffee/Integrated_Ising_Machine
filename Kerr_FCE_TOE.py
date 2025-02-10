# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 06:20:37 2025

@author: yanxi
"""

from scipy.integrate import complex_ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
#import cmath
import math
import time
from sympy import *
from sympy import symbols

def CMT3(t,Y,Delta,u,chi,tau,k_T,gamma_TPA,gamma_FCA,sigma_FCD, xi_T,eta_lin,eta_c,tao_theta):
    z = Y[0]
    n = Y[1]
    T = Y[2]
    dzdt = -(k_T/2+1j*(Delta+chi*abs(z)**2))*z-math.sqrt(k)*u 
    +1j*(n+sigma_FCD*n**0.8-T)*z - (gamma_TPA*abs(z)**2+gamma_FCA*n)*z
    dndt = abs(z)**4-n/tau
    dTdt = xi_T*abs(z)**2*(eta_lin*eta_c + 2*gamma_TPA*abs(z)**2 + 2*gamma_FCA*n) - T/tao_theta
    return [dzdt, dndt, dTdt]

chi = 0.55
tau=18.5
k_l=50 * abs(chi)
k = 150 * abs(chi)
k_T=k+k_l
Delta_th= -math.sqrt(3)/2.0 * k_T
gamma_TPA=0.11
gamma_FCA=0.2
sigma_FCD=7.2

xi_T=0.074;
eta_lin=0.4;
eta_c=1;
tao_theta=185;

# bistability
lower_limit = 0.0001
upper_limit = 3
num_pts = 100
E_list = np.linspace(lower_limit, upper_limit, num_pts)
Delta_range=np.arange(-300,100,50)
start = time.time()
for j in Delta_range:
    u_list=[]
    for i in E_list:
        #print('stable n is: {:.5f}'.format(18.5*abs(i)**2 ))
        # P = ((k_T/2 + gamma_TPA*i + gamma_FCA * (tau*i**2 ))**2 + (j + chi*i+ 
        #     tao_theta* xi_T*i*(eta_lin*eta_c+2*gamma_TPA*i + 2*gamma_FCA*tau*i**2) 
        #     -tau*i**2-sigma_FCD*(tau*i**2 )**0.8)**2)*i/k
        P = ((k_T/2 + gamma_TPA*i + gamma_FCA * tau*i**2)**2 + (j + chi*i+ 
            tao_theta* xi_T*i*(eta_lin*eta_c+2*gamma_TPA*i + 2*gamma_FCA*tau*i**2) 
            -tau*i**2-sigma_FCD*tau**0.8*i**1.6)**2)*i/k
        u_list.append(math.sqrt(P))
    plt.plot(u_list,E_list,label='$\Delta = {} $'.format(j))

end = time.time()
print('Time elapse for analytical method: {:.1f} min'.format((end - start)/60))
plt.title('Bifurcation Diagram of a Kerr Resonator with FCE and TOE')
plt.xlabel('$u$')
plt.ylabel('$|z|^2$')
#plt.xlim(0,100)
plt.legend()
#######################################
## plot bistability range 
plt.figure()
Delta_list=np.linspace(-510,0,50)
bi_u=[]
bi_delta=[]
E=symbols('E')
Delta=symbols('Delta') 
eqn=((k_T/2 + gamma_TPA*E + gamma_FCA * tau*E**2)**2 + (Delta + chi*E+ 
    tao_theta* xi_T*E*(eta_lin*eta_c+2*gamma_TPA*E + 2*gamma_FCA*tau*E**2) 
    -tau*E**2-sigma_FCD*tau**0.8*E**1.6)**2)*E/k
for i in Delta_list:
    u_list=[]
    E_diff1=[]
    for j in E_list:
        P=eqn.subs([(E,j),(Delta,i)])
        u_list.append(math.sqrt(P))
        new=eqn.diff(E).subs([(E,j),(Delta,i)])
        E_diff1.append(new)
    zero_crossings = np.where(np.diff(np.sign(E_diff1)))[0]+1
    #print(zero_crossings)
    if len(zero_crossings)==2:
        bi_delta.append(i)
        bi_u.append([u_list[i] for i in zero_crossings])
#print(len(bi_u),len(bi_u[0]))
plt.plot(bi_delta,[x[0] for x in bi_u],'g') # append all smallest u
plt.plot(bi_delta,[x[1] for x in bi_u],'g') # append all biggest u
plt.title('Bistability Diagram with Kerr, FCE and TOE')
plt.xlabel('$\Delta$')
plt.ylabel('u')


## As shown above, there should be split at around u=30.
## This time, the analytical method and numerical results are totally different. As u increase till 60, there was still no bistability.
# Delta=-300
# u=np.arange(0,400,30)  ## until u is around 150, there is bistability 
# y0 = [0+0j, 0+0j,0+0j]
# t0 = 0
# tfinal = 1500
# ts = np.linspace(t0, tfinal, 200)

# forward=[]
# backward=[]
# start = time.time()
# # from u=0 to high input power
# for k in u: 
#     parameter=[Delta,k,chi,tau,k_T,gamma_TPA,gamma_FCA,sigma_FCD, xi_T,eta_lin,eta_c,tao_theta]
#     # get stablilized z and n
#     ans=solve_ivp(CMT3,[t0, tfinal],y0,t_eval=ts,method='DOP853',args=parameter)
#     forward.append(abs(ans.y[0,-1])**2) # append the stabilized |z|**2
# end=ans.y[:,-1]
# print(end)
# # from high input power down to 0 
# for k in list(reversed(u)): 
#     parameter=[Delta,k,chi,tau,k_T,gamma_TPA,gamma_FCA,sigma_FCD, xi_T,eta_lin,eta_c,tao_theta]
#     # get stablilized z and n
#     ans=solve_ivp(CMT3,[t0, tfinal],end,t_eval=ts,method='DOP853',args=parameter)
    
#     backward.append(abs(ans.y[0,-1])**2) # append the stabilized |z|**2

# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Upward and Downward Integration')
# ax1.plot(u, forward,'b',label='Upward')  
# ax2.plot(list(reversed(u)), backward,'r',label='Downward')
# ax1.legend()
# ax2.legend()
# fig.supxlabel('$u$')
# fig.supylabel('$|z|^2$')

# fig_new,ax3=plt.subplots()
# fig_new.suptitle('Upward and Downward Integration')
# ax3.plot(u, forward,'b',label='Upward')  
# ax3.plot(list(reversed(u)), backward,'r',label='Downward')
# ax3.legend()
# fig_new.supxlabel('u')
# fig_new.supylabel('$|z|^2$')
# end = time.time()
# print('Time elapse for numerical method: {:.1f} min'.format((end - start)/60))