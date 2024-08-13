#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Implements classes and methods related to generating lightning strike waveforms.
"""
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
#from torchrf.constants import SPEED_OF_LIGHT, RETURN_STROKE_SPEED, EPSILON_NAUGHT

#from torchrf.rt.utils import random_uniform

PI = np.pi
SPEED_OF_LIGHT = 299792458
RETURN_STROKE_SPEED = 130000000 # Lightning return stroke speed from Chene et al.
DIELECTRIC_PERMITTIVITY_VACUUM = -1  # Fix this
EPSILON_NAUGHT = 8.854187817e-12

def random_uniform(shape,low,high,dtype):
    # Specify the range for the uniform distribution

    # Generate a random uniform tensor
    uniform_tensor = (high - low) * torch.rand(shape) + low
    uniform_tensor.type(dtype)
    return uniform_tensor

sigma = 0.7368 # Log-normal width from Petrov & Alessandro (1).

def get_polarity(): # Randomly picks a polarity for the lightning strike. 
    pol = 2.0*random.randint(0,1)-1.0
    return pol

def get_loc(x0,xf, y0,yf): # Generate a location for the lightning strike within a storm defined by dimenions x0 -> xf, y0 -> yf
    x = random.uniform(x0, xf)
    y = random.uniform(y0,yf)
    return (x, y)

def get_num_rs(): # Generate a Poisson distributed number of return strokes. 
    return np.random.poisson(3)

def f_i(x,x_max): # Funciton defining the distribution of return stroke currents around some maximum.
    a = 1/(np.sqrt(2*PI)*sigma*x) # Leading terms.
    a *= np.exp(-np.power(np.log(x)-np.log(x_max),2)/(2*np.power(sigma,2)))
    return a

def get_stroke_params(stroke_num): # Generates total current & other parameters in a given return stroke. Source: Petrov and Alessandro (1).
    stroke_pars = np.zeros(5)
    if stroke_num == 0:
        stroke_pars[0]=(100000) # I max - Amps
        stroke_pars[1]=(0.93) # Eta
        stroke_pars[2]=(19) # T1 - us
        stroke_pars[3]=(485) # T2 - us
        stroke_pars[4]=(10) # n
    else:
        stroke_pars[0]=(25000) # I max - Amps
        stroke_pars[1]=(0.993) # Eta
        stroke_pars[2]=(0.454) # T1 - us
        stroke_pars[3]=(143) # T2 - us
        stroke_pars[4]=(10) # n
    return stroke_pars

def get_I(numstrokes): # Samples the return stroke current distribtuion to generate stroke amplitude(s).
    i_stroke = np.zeros([numstrokes])
    p_i = np.zeros([1000])
    for j in range(numstrokes):
        I_bar = get_stroke_params(j)[0]
        i_width = I_bar*np.exp(-np.sqrt(2)*sigma)
        i_low = I_bar - 2.0 * i_width
        i_high = I_bar + 2.0 * i_width
        i_dist = np.linspace(i_low,i_high,1000) # Generates 1000 possible currents across 3 sigma in either direction around the median.
        for k in range(1000):
            p_i[k] = f_i(i_dist[k],I_bar)
        p_i /= p_i.sum()
        # print(p_i)
        i_stroke[j] = np.random.choice(i_dist,p=p_i)
    return i_stroke

def get_i_t_rs(stroke_num,I_stroke,T): # Returns the time-dependant current from a return stroke. t = 0 is assumed to be when a givens stroke starts.
    stroke_params = get_stroke_params(stroke_num)
    eta = stroke_params[1]
    tau_1 = stroke_params[2]
    tau_2 = stroke_params[3]
    n = stroke_params[4]
    #if stroke_num == 1:
        #print(T)
    if T > 0:
        i_t = I_stroke/eta * np.power(T/tau_1,n) / np.power(T/tau_1,n+1) * np.exp(-T/tau_2)
    else:
        i_t = 0
    return i_t

def get_stroke_timing(numstrokes): # Assumes t = 0 is the time of the stepped leader.
    t=0
    stroke_times = np.zeros([numstrokes])
    stroke_counter = 0 # Counts the number of return strokes that have happened.
    while (stroke_counter < numstrokes):
        survival_prob = np.exp(-t/133e-6) # Assumes exponential decay survival funciton and ~3 strokes/minute.
        rand_step = random.random()
        if rand_step > survival_prob:
            stroke_times[stroke_counter] = t
            t = 0
            stroke_counter += 1
        else:
            t += 1./20. # Go in steps of 10 us.
    if numstrokes > 1:
        for i in range(1,numstrokes):
            stroke_times[i] += stroke_times[i-1] # Makes strokes sequential.
    return stroke_times  

def get_E_rs(r,t,num_rs,stroke_I,stroke_times,pol): # Generate the E field of a return stroke at a distance r and time t based on Chen et al. (2).
    c = SPEED_OF_LIGHT
    v = RETURN_STROKE_SPEED
    beta = v / c
    h = beta * ( c * t - np.sqrt( np.power(beta*c*t,2)+np.power(r,2)*(1-np.power(beta,2)) )) / ( 1 - np.power(beta,2) )
    e_stat = 0 # Static E-field
    e_ind = 0 # Induction E-field
    e_rad = 0 # Radiated E-field
    for i in range(num_rs):
        i_t_stroke = get_i_t_rs(i,stroke_I[i],(t-stroke_times[i]))
       #if i == 1:
            #print(i_t_stroke)
        e_stat += pol[i]*i_t_stroke/(2*PI*EPSILON_NAUGHT) * ((-h*(t-stroke_times[i])+2*np.power(h,2)/v+np.power(r,2)/v)/\
                                                      np.power((np.power(h,2)+np.power(r,2)),3/2) - 1/(r*v)\
                                                      -1/(2*c*r)*(np.arctan(h/r)-3*h*r/(np.power(h,2)+np.power(r,2))))
        e_ind += pol[i]*i_t_stroke/(4*PI*EPSILON_NAUGHT*r*c) * (np.arctan(h/r)-3*h*r/(np.power(h,2)+np.power(r,2)))
        e_rad += pol[i]*i_t_stroke/(2*PI*EPSILON_NAUGHT*np.power(c,2)*np.power((np.power(h,2)+np.power(r,2)),3/2))*np.power(r,2)/\
        (1/v+h/(c*np.sqrt(np.power(h,2)+np.power(r,2))))
    e_tot = e_stat+e_ind+e_rad
    return e_tot

num_rs = get_num_rs()
print("Generating "+str(num_rs)+" return strokes.")
stroke_I = get_I(num_rs)
stroke_times = get_stroke_timing(num_rs)
pol = np.zeros([num_rs])
for j in range(num_rs):
    pol[j] = get_polarity()
    print("Polarity of stroke number "+str(j)+" is: "+str(pol[j]))
    print("Stroke "+str(j)+" occurs at t = "+str(stroke_times[j])+str(" s."))
    print("Stroke "+str(j)+" has a max current = "+str(stroke_I[j])+str(" Amps."))
x = np.zeros([1000000])
y = np.zeros([1000000])
for i in range(1000000):
	x[i] = float(i)*1e-6 # Each step is a microsecond.
	y[i] = get_E_rs(50000,x[i],num_rs,stroke_I,stroke_times,pol)

# plot
fig, ax = plt.subplots()
plt.figure(1)
ax.plot(x, y, linewidth=2.0)
sp = np.fft.rfft(y)
signalPSD = 10*np.log10(np.abs(sp) ** 2)
length = len(y)
timestep = x[1]-x[0]
freq = np.fft.rfftfreq(length, d=timestep)
plt.figure(2)
plt.plot(freq, signalPSD)

plt.show()

#########################################
################ Sources ################
#########################################
#
# (1) "Verification of lightning strike incidences as a Poisson process" - N. I. Petrov and F. D. Alessandro. 
#      Journal of Astmospheric and Solar-Terrestrial Physics 64 (2002) 1645-1650
#
# (2) "Approximate expressions for lightning electromagnetic fields at near and far ranges: Influence of return-stroke speed" - Chen, Wang, and Rakov.
#     Journal of Geophysical Research: Atmospheres. 2015.
#
#
#
#
#
#
#
#
#
#
#
#
