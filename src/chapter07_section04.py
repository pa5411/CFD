# -----------------------------------------------------------
# 1D Maccormack scheme for subsonic nozzle
# Author: pa5411
# -----------------------------------------------------------

# -----------------------------------------------------------
# libraries
# -----------------------------------------------------------

#numpy is used to store data
#pandas is used to store and display tabulated results 
#matplotlib is used to display graphical results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import math

# -----------------------------------------------------------
# user fixed settings
# -----------------------------------------------------------

NOZZLE_LENGTH = 3 
GAMMA = 1.4 #ratio of specific heats
TIME_STEPS = 5000
COURANT_NUMBER = 0.5
DX = 0.1 #grid spacing
PLOT_RESULTS = True #plot graphical data (boolean)
INT_PRECISION = np.int16
FLOAT_PRECISION = np.float32
EMPTY_TYPE = np.nan
ABS_TOLERANCE = 1e-15 #used for modulo comparisons

#specify pressure ratio for the subsonic case
PRESSURE_RATIO = 0.93

#specify iterations to plot data
TIME_PLOTS = [0, 50, 100, 150, 200, 700] 

# -----------------------------------------------------------
# generate 1D grid
# -----------------------------------------------------------

N = int(NOZZLE_LENGTH/DX + 1)
throat_loc = int(math.ceil(N/2)) - 1
print('Throat Location (index cell): ', throat_loc, '\n')

x_grid = np.linspace(
  0, 
  NOZZLE_LENGTH, 
  N, 
  dtype=FLOAT_PRECISION)

step_nos = np.linspace(
  0, 
  TIME_STEPS-1, 
  TIME_STEPS, 
  dtype=INT_PRECISION)

# -----------------------------------------------------------
# initial conditions for CFD
# -----------------------------------------------------------

D = 1 - 0.023*x_grid #density
T = 1 - 0.009333*x_grid #temperature
U = 0.05 + 0.11*x_grid #velocity

#array to store Area
A = np.empty(N, dtype=FLOAT_PRECISION)

for ii in range(N):
  if x_grid[ii] <= 1.5:
    multiplier = 2.2
  if x_grid[ii] > 1.5:
    multiplier = 0.2223
  #nozzle shape
  A[ii] = 1 + multiplier*np.power((x_grid[ii]-1.5),2) 

A_log = np.log(A)

# -----------------------------------------------------------
# arrays for storing CFD calculations 
# -----------------------------------------------------------

#array to store pressure from CFD
p = np.empty(N, dtype=FLOAT_PRECISION) #pressure

#array to store barred gradients from CFD
dD_dt_bar = np.empty(N, dtype=FLOAT_PRECISION)
dU_dt_bar = np.empty(N, dtype=FLOAT_PRECISION)
dT_dt_bar = np.empty(N, dtype=FLOAT_PRECISION)

#array to store corrected gradients from CFD
dD_dt_corr = np.empty(N, dtype=FLOAT_PRECISION)
dU_dt_corr = np.empty(N, dtype=FLOAT_PRECISION)
dT_dt_corr = np.empty(N, dtype=FLOAT_PRECISION)

#set arrays to defined empty type
p[:] = EMPTY_TYPE
dD_dt_bar[:] = EMPTY_TYPE
dU_dt_bar[:] = EMPTY_TYPE
dT_dt_bar[:] = EMPTY_TYPE
dD_dt_corr[:] = EMPTY_TYPE
dU_dt_corr[:] = EMPTY_TYPE
dT_dt_corr[:] = EMPTY_TYPE

# -----------------------------------------------------------
# arrays for storing CFD results 
# -----------------------------------------------------------

#store flow information at one grid point for each time step
D_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
U_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
T_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
p_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
M_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION) 
dD_dt_avg_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
dU_dt_avg_loc = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)

D_loc[:] = EMPTY_TYPE
U_loc[:] = EMPTY_TYPE
T_loc[:] = EMPTY_TYPE
p_loc[:] = EMPTY_TYPE
M_loc[:] = EMPTY_TYPE
dD_dt_avg_loc[:] = EMPTY_TYPE
dU_dt_avg_loc[:] = EMPTY_TYPE

results = [D_loc, T_loc, p_loc, M_loc]

#store mass flow rate at all grid points at a given time step
mdot_time = [] 

#run short code to obtain corrected index for time step 
#numbers:
time_plots_index = []
for kk in TIME_PLOTS:
  if kk == 0:
    time_plots_index.append(0)
  else:
    time_plots_index.append(kk-1)

# -----------------------------------------------------------
# set up analytical calculation for comparison with CFD
# -----------------------------------------------------------

#calculate exit Mach Number
M_ana_exit_sq = \
  (PRESSURE_RATIO**(-((GAMMA-1)/GAMMA)) - 1)  \
  / ((GAMMA-1)/2)
M_ana_exit = M_ana_exit_sq ** 0.5


#create valid Mach number inputs

'''
M_ana_start = 0.1 #Mach nos. at x=0
M_ana_end = 3.35 #Mach nos. at x=3
M_ana_N = int((M_ana_end-M_ana_start)/0.001) + 1
M_ana = np.linspace(
  M_ana_start, 
  M_ana_end, 
  M_ana_N, 
  dtype=FLOAT_PRECISION) 
'''

#create x points with which analytical data is stored

'''
x_ana = np.empty(M_ana_N, dtype=FLOAT_PRECISION)
x_ana[:] = EMPTY_TYPE
'''

# -----------------------------------------------------------
# analytical calculation
# -----------------------------------------------------------

#calculate normalised area
'''
_M_temp = 1 + 0.5*(GAMMA-1)*M_ana**2
_M_exp = (GAMMA+1)/(GAMMA-1)
_A_ana_sq = (1/M_ana**2) * ((2/(GAMMA+1)) * _M_temp)**_M_exp
_A_ana = _A_ana_sq**0.5
'''

#calculate x points
'''
for ii, M_A in enumerate(M_ana):
  if M_A <= 1:
    x_ana[ii] = -(np.power(((_A_ana[ii] - 1)/2.2),0.5)) + 1.5
  else:
    x_ana[ii] = np.power(((_A_ana[ii] - 1)/2.2),0.5) + 1.5
'''

#calculate flow variables
'''
p_ana = _M_temp**(-GAMMA/(GAMMA-1))
D_ana = _M_temp**(-1/(GAMMA-1))
T_ana = _M_temp**-1
'''

#remove results which fall outside of valid domain of x
'''
M_ana = M_ana[x_ana >= 0] 
p_ana = p_ana[x_ana >= 0]
D_ana = D_ana[x_ana >= 0] 
T_ana = T_ana[x_ana >= 0] 
x_ana = x_ana[x_ana >= 0]

results_ana = [M_ana, p_ana, D_ana, T_ana]
'''

# -----------------------------------------------------------
# maccormack CFD scheme
# -----------------------------------------------------------

for jj in range(TIME_STEPS):

  #predictor step - calculate predicted gradients at internal
  #points
  for ii in range(1,N-1):

    dD_dt_bar[ii] = \
      -D[ii] * ((U[ii+1] - U[ii])/DX) \
      -D[ii] * U[ii] * ((A_log[ii+1] - A_log[ii])/DX) \
      -U[ii] * ((D[ii+1] - D[ii])/DX)

    dU_dt_bar[ii] = \
      -U[ii] * ((U[ii+1] - U[ii])/DX) \
      -(1/GAMMA) \
      *( ((T[ii+1] - T[ii])/DX) \
        +(T[ii]/D[ii]) * ((D[ii+1] - D[ii])/DX) \
        )

    dT_dt_bar[ii] = \
      -U[ii] * ((T[ii+1] - T[ii])/DX) \
      -(GAMMA-1)*T[ii] \
      *( \
        ((U[ii+1] - U[ii])/DX) \
        +U[ii] * ((A_log[ii+1] - A_log[ii])/DX) \
        )

  #calculate minimum time step
  a = np.power(T,0.5) #speed of sound
  delta_t = np.nanmin(COURANT_NUMBER*DX/(a + U))

  #predictor - calculate barred quantities at internal points
  D_bar = D + dD_dt_bar*delta_t
  U_bar = U + dU_dt_bar*delta_t
  T_bar = T + dT_dt_bar*delta_t

  #assign values to external points of predicted variables 
  #at inflow for backwards difference scheme 
  D_bar[0] = D[0]
  U_bar[0] = U[0]
  T_bar[0] = T[0]

  #corrector - calculate corrected gradients
  for ii in range(1,N-1):
    dD_dt_corr[ii] = \
      -D_bar[ii] * (U_bar[ii] - U_bar[ii-1]) * (1/DX) \
      -D_bar[ii] * U_bar[ii] * (A_log[ii] - A_log[ii-1]) \
      * (1/DX) \
      -U_bar[ii] * (D_bar[ii] - D_bar[ii-1]) * (1/DX)

    dU_dt_corr[ii] = \
      -U_bar[ii] * (U_bar[ii] - U_bar[ii-1]) * (1/DX) \
      -(1/GAMMA) * (T_bar[ii] - T_bar[ii-1]) * (1/DX) \
      -(1/GAMMA) * (T_bar[ii]/D_bar[ii]) \
        *(D_bar[ii] - D_bar[ii-1]) * (1/DX)

    dT_dt_corr[ii] = \
      -U_bar[ii] * (T_bar[ii] - T_bar[ii-1]) * (1/DX) \
      -(GAMMA-1) * T_bar[ii] * (U_bar[ii] - U_bar[ii-1]) \
      * (1/DX) \
      -(GAMMA-1) * T_bar[ii] * U_bar[ii] *(1/DX) \
        *(A_log[ii] - A_log[ii-1]) 

  #calculate average time derivatives
  dD_dt_avg = (dD_dt_bar + dD_dt_corr)*0.5
  dU_dt_avg = (dU_dt_bar + dU_dt_corr)*0.5
  dT_dt_avg = (dT_dt_bar + dT_dt_corr)*0.5

  #update variables 
  D = D + dD_dt_avg*delta_t
  U = U + dU_dt_avg*delta_t
  T = T + dT_dt_avg*delta_t
  p = D*T

  #set boundary conditions
  D[0] = 1 #fixed inflow density
  T[0] = 1 #fixed inflow temperature
  p[0] = D[0]*T[0] #derived inflow pressure
  U[0] = 2*U[1] - U[2] #floating inflow velocity 
  p[N-1] = PRESSURE_RATIO #derived outflow pressure
  D[N-1] = 2*D[N-2] - D[N-3] #floating outflow density
  U[N-1] = 2*U[N-2] - U[N-3] #floating outflow velocity
  T[N-1] = p[N-1]/D[N-1] #floating outflow temperature

  #store results at one location per time step
  D_loc[jj] = D[throat_loc]
  U_loc[jj] = U[throat_loc]
  T_loc[jj] = T[throat_loc]
  p_loc[jj] = p[throat_loc]
  M_loc[jj] = U[throat_loc]/np.power(T[throat_loc],0.5)
  dD_dt_avg_loc[jj] = np.absolute(dD_dt_avg[throat_loc])
  dU_dt_avg_loc[jj] = np.absolute(dU_dt_avg[throat_loc])

  #store results along nozzle at a given time step
  if jj in time_plots_index:
      mdot_time.append(D*A*U)


# -----------------------------------------------------------
# Post-Processing
# -----------------------------------------------------------

#calculate other results
M = U/np.power(T,0.5)
mdot = D*A*U

# -----------------------------------------------------------
# Results - Table 7.7
# -----------------------------------------------------------

df = pd.DataFrame(
    {'x': x_grid.tolist(),
     'area': A.tolist(),
     'density': D.tolist(),
     'velocity': U.tolist(),
     'temp': T.tolist(),
     'pressure': p.tolist(),
     'mach': M.tolist(),
     'mdot': mdot.tolist()
    })

df.index = np.arange(1, len(df) + 1)
title_table_7_7 = \
  'Table 7.7 - Results across nozzle after {} steps' \
    .format(TIME_STEPS)
print(title_table_7_7)
print(df)



