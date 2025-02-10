# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:44:20 2025

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

### global values
chi = 0.55
tau=18.5
k_l=50 * abs(chi)
k = 150 * abs(chi)
k_T=k+k_l

### analytical 

lower_limit = 0.0001 # not zero to avoid nan
upper_limit = 600
num_pts = 100
E_list = np.linspace(lower_limit, upper_limit, num_pts)

Delta_range=np.arange(-300,100,50)

E=symbols('E')
Delta=symbols('Delta')
eqn=((k_T/2)**2 + (Delta + chi*E)**2)*E/k
E_diff1=[]

############ plot bifurcation diagram illustrating bistability
# for j in Delta_range:
#     u_list=[] ## temporary list to save all the u value for each Delta
#     for i in E_list:
#         P=eqn.subs([(E,i),(Delta,j)])
#         u_list.append(math.sqrt(P))
#     plt.plot(u_list,E_list,label='$\Delta = {} $'.format(j))
# plt.title('Bifurcation Diagram of a Single Kerr Resonator')
# plt.xlabel('$u$')
# plt.ylabel('$|z|^2$')
# plt.legend()
##########################
## choose a working point where Delta=-200, bistability happens around u=180
# plt.figure()
# Delta_op=-200
# u_list_op=[]  ## if not renamed, remember to empty the u_list!
# E_diff1=[]
# for i in E_list:
#     P=eqn.subs([(E,i),(Delta,Delta_op)])
#     u_list_op.append(math.sqrt(P))
#     # 1. order differential 
#     new=eqn.diff(E).subs([(E,i),(Delta,Delta_op)])
#     E_diff1.append(new)
# # plot the curve E vs u under the working point Delta=-200 
# # to see where there are bistable states
# plt.plot(u_list_op,E_list,label='$\Delta = {} $'.format(Delta_op))
# plt.title('Bifurcation Diagram of a Single Kerr Resonator')
# # plot the input range for bistability 
# # index where the Derivative change sign (approximately where f'(x)=0)
# zero_crossings = np.where(np.diff(np.sign(E_diff1)))[0]+1
# # find the boundries of u 
# u_range=[u_list_op[i] for i in zero_crossings]
# for i in u_range:
#     plt.axvline(x=i,color='purple', linestyle='dashed')
    
# plt.xlabel('$u=\sqrt{P}$')
# plt.ylabel('$E=|z|^2$')
# plt.legend()

#######################################
## plot bistability range 
plt.figure()
Delta_list=np.linspace(-510,0,50)
bi_u=[]
bi_delta=[]
# E=symbols('E')
# Delta=symbols('Delta') 
# eqn=((k_T/2)**2 + (Delta + chi*E)**2)*E/k
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
plt.title('Bistability Diagram with Kerr Only')
plt.xlabel('$\Delta$')
plt.ylabel('u')

# ### numerical 
# plt.figure()
def CMT1(t,z,Delta,u,chi,tau,k_T,k):
    dzdt = -(k_T/2+1j*(Delta+chi*abs(z)**2))*z-math.sqrt(k)*u
    return dzdt

Delta=-200
u=200
z0=[0+0j]
t_span=(0,0.5)  ## from the plot we can see that it converges within 0.5

### plot time development of z, to see how long it takes to converge
### this time will be required in the t_span in numerical solver
# parameter=[Delta,u,chi,tau,k_T,k]
# ans=solve_ivp(CMT1,t_span,z0,args=parameter)
# plt.plot(ans.t, ans.y[0].real,label='Re{z}')
# plt.plot(ans.t, ans.y[0].imag,label='Im{z}')
# plt.xlabel('t')
# plt.legend()

#### plot switching like bistability
# forward=[]
# backward=[]
# start = time.time()
# u_list=np.linspace(0, 201,20)
# # from u=0 to high input power
# for i in u_list: 
#     parameter=[Delta,i,chi,tau,k_T,k]
#     # get stablilized z and n
#     ans=solve_ivp(CMT1,t_span,z0,args=parameter)
#     #print('Upward stable z is: {}'.format(ans.y[0,-1]))
#     forward.append(abs(ans.y[0,-1])**2) # append the stabilized |z|**2
# end=ans.y[:,-1]
# #print(end)
# # from high input power down to 0, initial point is with high z and n, e.g. the end of upward or (100,100)
# for j in list(reversed(u_list)): 
#     parameter=[Delta,j,chi,tau,k_T,k]
#     # get stablilized z and n
#     ans=solve_ivp(CMT1,t_span,end,args=parameter) 
#     #print(ans.y)
#     #print('Downward stable z is: {}'.format(ans.y[0,-1]))
#     backward.append(abs(ans.y[0,-1])**2) # append the stabilized |z|**2

# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Upward and Downward Integration')
# ax1.plot(u_list, forward,'b',label='Upward')  
# ax2.plot(list(reversed(u_list)), backward,'r',label='Downward')
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper left")
# fig.supxlabel('u')
# fig.supylabel('$|z|^2$')
# plt.show()

# plt.figure()
# plt.title('Upward and Downward Integration')
# plt.plot(u_list, forward,'b',label='Upward')  
# plt.plot(list(reversed(u_list)), backward,'r',label='Downward')
# plt.xlabel('u')
# plt.ylabel('$|z|^2$')
# plt.legend()
# plt.show()

# finish = time.time()
# print('Time elapse: {:.1f} min'.format((finish - start)/60))