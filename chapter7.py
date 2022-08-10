import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#User fixed settings 
NOZZLE_LENGTH = 3 #centimeters
GAMMA = 1.4
TIME_STEPS = 1400
COURANT_NUMBER = 0.5
DX = 0.1
INT_PRECISION = np.int16
FLOAT_PRECISION = np.float32
EMPTY_TYPE = np.nan
LOC = 15 #grid location to plot results

#generate 1D grid + number of steps
N = int(NOZZLE_LENGTH/DX + 1)
x_grid = np.linspace(0, NOZZLE_LENGTH, N, dtype=FLOAT_PRECISION)
step_nos = np.linspace(0, TIME_STEPS-1, TIME_STEPS, dtype=INT_PRECISION)

#initial conditions
D = 1 - 0.3146*x_grid #density
T = 1 - 0.2314*x_grid #temperature
U = (0.1 + 1.09*x_grid)*np.power(T,0.5) #velocity
A = 1 + 2.2*np.power((x_grid-1.5),2) #nozzle shape
A_log = np.log(A)

#create arrays
p = np.empty(N, dtype=FLOAT_PRECISION) #pressure

dD_dt_bar = np.empty(N, dtype=FLOAT_PRECISION)
dU_dt_bar = np.empty(N, dtype=FLOAT_PRECISION)
dT_dt_bar = np.empty(N, dtype=FLOAT_PRECISION)

dD_dt_corr = np.empty(N, dtype=FLOAT_PRECISION)
dU_dt_corr = np.empty(N, dtype=FLOAT_PRECISION)
dT_dt_corr = np.empty(N, dtype=FLOAT_PRECISION)

p[:] = EMPTY_TYPE
dD_dt_bar[:] = EMPTY_TYPE
dU_dt_bar[:] = EMPTY_TYPE
dT_dt_bar[:] = EMPTY_TYPE
dD_dt_corr[:] = EMPTY_TYPE
dU_dt_corr[:] = EMPTY_TYPE
dT_dt_corr[:] = EMPTY_TYPE

#create arrays to store normalised results
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

#begin maccormack scheme
for jj in range(TIME_STEPS):

  #predictor step - calculate predicted gradients at internal points
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

  #assign values to external points of predicted variables at inflow 
  #for backwards difference scheme 
  D_bar[0] = D[0]
  U_bar[0] = U[0]
  T_bar[0] = T[0]

  #corrector - calculate corrected gradients
  for ii in range(1,N-1):
    dD_dt_corr[ii] = \
      -D_bar[ii] * (U_bar[ii] - U_bar[ii-1]) * (1/DX) \
      -D_bar[ii] * U_bar[ii] * (A_log[ii] - A_log[ii-1]) * (1/DX) \
      -U_bar[ii] * (D_bar[ii] - D_bar[ii-1]) * (1/DX)

    dU_dt_corr[ii] = \
      -U_bar[ii] * (U_bar[ii] - U_bar[ii-1]) * (1/DX) \
      -(1/GAMMA) * (T_bar[ii] - T_bar[ii-1]) * (1/DX) \
      -(1/GAMMA) * (T_bar[ii]/D_bar[ii]) \
        *(D_bar[ii] - D_bar[ii-1]) * (1/DX)

    dT_dt_corr[ii] = \
      -U_bar[ii] * (T_bar[ii] - T_bar[ii-1]) * (1/DX) \
      -(GAMMA-1) * T_bar[ii] * (U_bar[ii] - U_bar[ii-1]) * (1/DX) \
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
  D[0] = 1
  T[0] = 1
  p[0] = D[0]*T[0] 
  U[0] = 2*U[1] - U[2] #floating inflow B.C
  D[N-1] = 2*D[N-2] - D[N-3] #floating outflow B.C
  U[N-1] = 2*U[N-2] - U[N-3] #floating outflow B.C
  T[N-1] = 2*T[N-2] - T[N-3] #floating outflow B.C
  p[N-1] = D[N-1]*T[N-1]

  #store results
  D_loc[jj] = D[LOC]
  U_loc[jj] = U[LOC]
  T_loc[jj] = T[LOC]
  p_loc[jj] = p[LOC]
  M_loc[jj] = U[LOC]/np.power(T[LOC],0.5)
  dD_dt_avg_loc[jj] = np.absolute(dD_dt_avg[LOC])
  dU_dt_avg_loc[jj] = np.absolute(dU_dt_avg[LOC])

#plotting 
#figure 1
fig, axs = plt.subplots(4)
fig.suptitle('Fig 7.9')
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

for kk in range(4):
  axs[kk].scatter(step_nos, results[kk], s=1, color='black')
  axs[kk].set_xlim(0, TIME_STEPS)

  if kk == 0:
    axs[kk].set_ylim(0.5,1)
    axs[kk].set_ylabel(r"$\frac{D}{D_o}$", rotation=0)
  elif kk == 1:
    axs[kk].set_ylim(0.6,1)
    axs[kk].set_ylabel(r"$\frac{T}{T_o}$", rotation=0)
  elif kk == 2:
    axs[kk].set_ylim(0.3,1)
    axs[kk].set_ylabel(r"$\frac{P}{P_o}$", rotation=0)
  else:
    axs[kk].set_ylim(0.8,1.8)
    axs[kk].set_ylabel(r"$M$", rotation=0)
    axs[kk].set_xlabel("Number of Time Steps")

  axs[kk].yaxis.set_label_coords(0.9, 0.6)

#figure 2
g = plt.figure(2)
#plt.scatter(step_nos, dD_dt_avg_loc, s=1, color='black')
plt.scatter(step_nos, dU_dt_avg_loc, s=2, color='red')
#plt.scatter(step_nos, dT_dt_avg_loc, s=1, color='red')
plt.xlabel("Number of Time Steps")
plt.ylabel("Residuals")
plt.yscale('log')

plt.show()
#np.savetxt(r'test1.txt', dD_dt_avg_loc, delimiter=",")

#pandas datatable 
df = pd.DataFrame(
    {'x': x_grid.tolist(),
     'area': A.tolist(),
     'density': D.tolist(),
     'velocity': U.tolist(),
     'temp': T.tolist(),
     'pressure':  p.tolist(),
     'mach': (U/np.power(T,0.5)).tolist()
    })

df.index = np.arange(1, len(df) + 1)
print(df)