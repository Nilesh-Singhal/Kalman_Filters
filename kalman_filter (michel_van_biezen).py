# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:43:38 2021

@author: niles
"""


"""Michel Van Biezen KF Example"""

#1D Plane Position and velocity Tracking (X-axis) with constant acceleration
"""
Given
vo x = 280 m/s
xo = 4000 m

OBSERVATION
xo = 4000 m     vo x = 280 m/s   
x1 = 4260 m     v1 x = 282 m/s
x2 = 4550 m     v2 x = 285 m/s
x3 = 4860 m     v3 x = 286 m/s
x4 = 5110 m     v4 x = 290 m/s

INITIAL CONDITIONS
ax = 2 m/s^2
vx = 280 m/s  
delta t = 1 sec
delta x = 25 m

PROCESS ERRORS 
sigma x = 20 m
sigma v = 5 m/s

OBSERVATION ERRORS
sigma x = 25 m
sigma v = 6 m/s

TRANSFORMATION MATRIX
A = [[1  delta t],[0  1]]
B = [[1/2*delta t^2],[delta t]]
C = [[1 0],[0 1]]
H = [[1 0],[0 1]]

NOISE ERRORS (Noise)
Assumed
state error W = 0
measurement error Z = 0

COVARIANCE MATRIX
P = State Covariance Matrix (Error in the estimate)
Q = Process Noise Covariance Matrix   = 0 (Assumed)
R = Measurement Covariance Matrix (Error in the measurement)

Assumed
P = [[sigma x^2    sigma x*sigma v],[sigma x*sigma v   sigma v^2]]
R = [[sigma x^2    sigma x*sigma v],[sigma x*sigma v   sigma v^2]]

INITIAL STATE MATRIX
X = [[xo],[vo x]]

INITIAL CONTROL VARIABLE (in this case it is acceleration which is constant 
for this example)
mu = Initial acceleration = ax = 2 m/s^2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predicted_state(current_state_matrix, control_variable, W):
    predicted_state_matrix = A*current_state_matrix + B*control_variable + W
    return predicted_state_matrix

def initial_process_covariance_matrix(σxp, σvp):
    ipcm = [[σxp**2,σxp*σvp],[σxp*σvp,σvp**2]]
    ipcm = np.matrix(ipcm)
    #it is assumed that σx and σv ar independent thus σxσv is taken as 0
    ipcm[0,1] = 0;ipcm[1,0] = 0
    return ipcm

def predicted_process_covariance_matrix(ipcm, Q):
    ppcm = A*ipcm*A.T + Q
    #it is assumed that σx and σv ar independent thus σxσv is taken as 0
    ppcm[0,1] = 0;ppcm[1,0] = 0
    return ppcm
    
def measurement_covariance_matrix(σxm, σvm):
    mcm = [[σxm**2,σxm*σvm],[σxm*σvm,σvm**2]]
    mcm = np.matrix(mcm)
    #it is assumed that σx and σv ar independent thus σxσv is taken as 0
    mcm[0,1] = 0;mcm[1,0] = 0
    return mcm

def kalman_gain(ppcm, mcm):
    KG = (ppcm*H.T)/(H*ppcm*H.T + mcm)
    KG = np.nan_to_num(KG)
    return KG

def new_measurement(state_measurement, Z):
    n_measurement = C*state_measurement + Z
    return n_measurement
    
def calculating_current_state(predicted_state_matrix, KG, n_measurement):
    calculated_current_state = predicted_state_matrix + KG*(n_measurement - H*predicted_state_matrix)
    return calculated_current_state

def updated_process_covariance_matrix(KG, ppcm):
    upcm = (np.eye(2) - KG*H)*ppcm
    return upcm

A = [[1, 1],[0, 1]]
B = [1/2,1]
C = [[1, 0],[0, 1]]
H = [[1, 0],[0, 1]]
#X = [4000,280]
A = np.matrix(A)
B = np.matrix(B).T
C = np.matrix(C)
H = np.matrix(H)
#current_state_matrix = np.matrix(X).T
control_variable = 2
W = 0
Q = 0
Z = 0

#process error
σxp = 20
σvp = 5 

#observation error
σxm = 25
σvm = 6 

#total number of obsrvations = 5
n_observation = 5
x0 = [4000,280]
x1 = [4260,282]
x2 = [4550,285]
x3 = [4860,286]
x4 = [5110,290]
x_tuple = (x0, x1, x2, x3, x4)
observations = np.vstack(x_tuple).T    

predicted_state_matrix = np.zeros((4, 2, 1))
n_measurement = np.zeros((4, 2, 1))
calculated_current_state = np.zeros((4, 2, 1)) 
KG = np.zeros((4,2,2))
ppcm = np.zeros((4,2,2))
upcm = np.zeros((4,2,2))

"""First Iteration"""
predicted_state_matrix[0,:,:] = predicted_state(observations[0:2, 0:1], control_variable, W)
ipcm= initial_process_covariance_matrix(σxp, σvp)
ppcm[0,:,:] = predicted_process_covariance_matrix(ipcm, Q)
mcm = measurement_covariance_matrix(σxm, σvm)
KG[0,:,:] = kalman_gain(ppcm[0,:,:], mcm)
state_measurement = observations[0:2, 1:2]
n_measurement[0,:,:] = new_measurement(state_measurement, Z)
calculated_current_state[0,:,:] = calculating_current_state(predicted_state_matrix[0,:,:], KG[0,:,:], n_measurement[0,:,:])
upcm[0,:,:] = updated_process_covariance_matrix(KG[0,:,:], ppcm[0,:,:])

"""Subsequent Iterations"""
for i in range(0,n_observation - 2):
    predicted_state_matrix[i+1,:,:] = predicted_state(calculated_current_state[i,:,:], control_variable, W)
    ppcm[i+1,:,:] = predicted_process_covariance_matrix(upcm[i,:,:], Q)
    KG[i+1,:,:] = kalman_gain(ppcm[i+1,:,:], mcm)
    state_measurement = observations[0:2, i+2:i+3]
    n_measurement[i+1,:,:] = new_measurement(state_measurement, Z)
    calculated_current_state[i+1,:,:] = calculating_current_state(predicted_state_matrix[i+1,:,:], KG[i+1,:,:], n_measurement[i+1,:,:])
    upcm[i+1,:,:] = updated_process_covariance_matrix(KG[i+1,:,:], ppcm[i+1,:,:])


fw = 10 # figure width

plt.figure(figsize=(fw,5))    
plt.plot((1,2,3,4), calculated_current_state[0:4,0:1,0:1].T.reshape(4,1), label='KALMAN PREDICTIONS')
plt.plot((1,2,3,4), predicted_state_matrix[0:4,0:1,0:1].T.reshape(4,1), label= 'PREDICTED STATE')
plt.plot((0,1,2,3,4), observations[0:1,:].T, label='SENSOR MEASUREMENT')
plt.xlim(1, 4);
plt.legend(loc='best');
plt.title('Tracking Position');
plt.show()

plt.figure(figsize=(fw,5))    
plt.plot((1,2,3,4), calculated_current_state[0:4,1:2,0:1].T.reshape(4,1), label='KALMAN PREDICTIONS')
plt.plot((1,2,3,4), predicted_state_matrix[0:4,1:2,0:1].T.reshape(4,1), label= 'PREDICTED STATE')
plt.plot((0,1,2,3,4), observations[1:2,:].T, label='SENSOR MEASUREMENT')
plt.xlim(1, 4);
plt.legend(loc='best');
plt.title('Tracking Velocity');
plt.show()

