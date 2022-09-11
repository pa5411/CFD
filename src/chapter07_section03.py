# -----------------------------------------------------------
# 1D Maccormack scheme for supersonic-subsonic nozzle
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
TIME_STEPS = 1400
COURANT_NUMBER = 0.5
DX = 0.1 #grid spacing
INT_PRECISION = np.int16
FLOAT_PRECISION = np.float32
EMPTY_TYPE = np.nan

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

D = 1 - 0.3146*x_grid #density
T = 1 - 0.2314*x_grid #temperature
U = (0.1 + 1.09*x_grid)*np.power(T,0.5) #velocity
A = 1 + 2.2*np.power((x_grid-1.5),2) #nozzle shape
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

#create valid Mach number inputs
M_ana_start = 0.1 #Mach nos. at x=0
M_ana_end = 3.35 #Mach nos. at x=3
M_ana_N = int((M_ana_end-M_ana_start)/0.01) + 1
M_ana = np.linspace(
  M_ana_start, 
  M_ana_end, 
  M_ana_N, 
  dtype=FLOAT_PRECISION) 

#create x points with which analytical data is stored
x_ana = np.empty(M_ana_N, dtype=FLOAT_PRECISION)
x_ana[:] = EMPTY_TYPE

# -----------------------------------------------------------
# analytical calculation
# -----------------------------------------------------------

#calculate normalised area
_M_temp = 1 + 0.5*(GAMMA-1)*M_ana**2
_M_exp = (GAMMA+1)/(GAMMA-1)
_A_ana_sq = (1/M_ana**2) * ((2/(GAMMA+1)) * _M_temp)**_M_exp
_A_ana = _A_ana_sq**0.5

#calculate x points
for ii, M_A in enumerate(M_ana):
  if M_A <= 1:
    x_ana[ii] = -(np.power(((_A_ana[ii] - 1)/2.2),0.5)) + 1.5
    if M_A == 1:
      throat_ana_index = ii
  else:
    x_ana[ii] = np.power(((_A_ana[ii] - 1)/2.2),0.5) + 1.5

#calculate flow variables
p_ana = _M_temp**(-GAMMA/(GAMMA-1))
D_ana = _M_temp**(-1/(GAMMA-1))
T_ana = _M_temp**-1

results_ana = [M_ana, p_ana, D_ana, T_ana]

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
  D[N-1] = 2*D[N-2] - D[N-3] #floating outflow density
  U[N-1] = 2*U[N-2] - U[N-3] #floating outflow velocity
  T[N-1] = 2*T[N-2] - T[N-3] #floating outflow temperature
  p[N-1] = D[N-1]*T[N-1] #derived outflow pressure

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

#estimate percentage differences

# -----------------------------------------------------------
# Results - Table 7.3
# -----------------------------------------------------------

df = pd.DataFrame(
    {'x': x_grid.tolist(),
     'area': A.tolist(),
     'density': D.tolist(),
     'velocity': U.tolist(),
     'temp': T.tolist(),
     'pressure':  p.tolist(),
     'mach': (U/np.power(T,0.5)).tolist(),
     'mdot': (D*A*U).tolist()
    })

df.index = np.arange(1, len(df) + 1)
print('Table 7.3')
print(df)

# -----------------------------------------------------------
# Results - Table 7.4
# -----------------------------------------------------------

df_ana_num_diff = pd.DataFrame(
    {'x': x_grid.tolist(),
     'area': A.tolist(),
     'density': D.tolist(),
     'mach': (U/np.power(T,0.5)).tolist(),
    })

df_ana_num_diff.index = np.arange(1, len(df) + 1)
print()
print('Table 7.4')
print(df_ana_num_diff)
print()

# -----------------------------------------------------------
# Results - Table 7.5
# -----------------------------------------------------------

df_grid_dep = pd.DataFrame(
    {'density': D[throat_loc],
     'temp': T[throat_loc],
     'pressure':  p[throat_loc],
     'mach': (U[throat_loc]/np.power(T[throat_loc],0.5))
    },
    index=[0])

print()
print('Table 7.5')
print(df_grid_dep)
print()

# -----------------------------------------------------------
# Results - Figure 7.2
# -----------------------------------------------------------

fig_ana, axs_ana = plt.subplots(4)
fig_ana.suptitle('Fig 7.2')
fig_ana.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

throat_index = np.where(x_ana==1.5)
results_ana_labels = [
  "M",
  r"$\frac{p}{p_o}$",
  r"$\frac{D}{D_o}$",
  r"$\frac{T}{T_o}$"]                    

for ii in range(4):
  axs_ana[ii].scatter(x_ana, results_ana[ii], s=1, color='k')
  axs_ana[ii].axvline(1.5, color='r', linestyle='-.')
  
  throat_value = results_ana[ii][throat_index]
  axs_ana[ii].axhline(
    throat_value, 
    color='r', 
    linestyle='-.')

  #https://stackoverflow.com/questions/42877747/
  #plotting additional y point on y axis
  trans = transforms.blended_transform_factory(
    axs_ana[ii].get_yticklabels()[0].get_transform(), 
    axs_ana[ii].transData)
  
  axs_ana[ii].text(
    0,
    float(throat_value), 
    "{:.3f}".format(float(throat_value)), 
    color="red", 
    transform=trans, 
    ha="right", 
    va="center")

  #labels
  axs_ana[0].set_ylabel(results_ana_labels[ii], rotation=0)
  if ii == 0:
    axs_ana[ii].yaxis.set_label_coords(0.9, 0.3)
  else:
    axs_ana[ii].yaxis.set_label_coords(0.9, 0.6)

axs_ana[3].set_xlabel("x")

# -----------------------------------------------------------
# Results - Figure 7.9
# -----------------------------------------------------------

fig, axs = plt.subplots(4)
fig.suptitle('Fig 7.9')
fig.subplots_adjust(left=0.1,
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

# -----------------------------------------------------------
# Results - Figure 7.10
# -----------------------------------------------------------

g1 = plt.figure(3)
g1.suptitle('Fig 7.10')
#plt.scatter(step_nos, dD_dt_avg_loc, s=1, color='black')
plt.scatter(step_nos, dU_dt_avg_loc, s=2, color='red')
#plt.scatter(step_nos, dT_dt_avg_loc, s=1, color='red')
plt.xlabel("Number of Time Steps")
plt.ylabel("Residuals")
plt.yscale('log')

#np.savetxt(r'test1.txt', dD_dt_avg_loc, delimiter=",")

# -----------------------------------------------------------
# Results - Figure 7.11
# -----------------------------------------------------------

markers = ['ko','k>','ks','kx','k+','k*']
g2 = plt.figure(4)
g2.suptitle('Fig 7.11')

for ii in range(6):
  plt.plot(x_grid, mdot_time[ii], markers[ii])

plt.xlabel("x/L")
plt.ylabel("mdot")

#plot analytical result as a horizontal line - this uses a 
#formula which is only valid at the nozzle throat
plt.axhline(
  D_ana[throat_ana_index]*T_ana[throat_ana_index]**0.5,
  color='red',
  dashes=(5, 2, 1, 2),
  label="Extra label on the legend") 

#plot legend
legend_items = \
  ["Time Step = " + str(jj) for jj in TIME_PLOTS] \
  + ['Analytical']
plt.legend(
  legend_items, 
  loc="best")

# -----------------------------------------------------------
# Results - Figure 7.12
# -----------------------------------------------------------

g3, axs_left = plt.subplots()
g3.suptitle('Fig 7.12')

axs_right = axs_left.twinx()
axs_left.set_ylabel(r"$\frac{D}{D_o}$", rotation=0)
axs_right.set_ylabel("M", rotation=0)
plt.xlabel("x/L")

#determine styling for the arrow drawn on figure
arrow_style = dict(
    arrowstyle = '<|-',
    color = 'black',
    linewidth = 1,
    linestyle = '-',
    mutation_scale = 20)

#create arrow annotations
axs_left.annotate(
  '', 
  xy = (1.5,0.6), 
  xytext = (1.1,0.6),
  arrowprops = arrow_style)

axs_right.annotate(
  '', 
  xy = (2.37,2.4), 
  xytext = (2.77,2.4),
  arrowprops = arrow_style)   

#identify indicies with values closest to a given 1dp number
idx_results_ana = []
for ii in np.arange(0.0,3.3,0.3):
  idx_results_ana.append((np.abs(x_ana - ii)).argmin())

axs_left.plot(
  x_ana[idx_results_ana], 
  D_ana[idx_results_ana], 
  'ro')
  
axs_right.plot(
  x_ana[idx_results_ana], 
  M_ana[idx_results_ana], 
  'ro')

#plot numerical result
axs_left.plot(x_grid, D, 'k-') #density
axs_right.plot(x_grid,U/(T**0.5),'k-') #mach number

#plot legend
legend_items = ['Exact analytical value', 'Numerical result']
plt.legend(
  legend_items, 
  loc="best")

plt.show()