import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#User fixed settings 
NOZZLE_LENGTH_PRIME = 3 #centimeters
GAMMA = 1.4
TIME_STEPS = 1400
COURANT_NUMBER = 0.25
GRID_SPACING = 0.1
INT_PRECISION = np.int16
FLOAT_PRECISION = np.float16
EMPTY_TYPE = np.nan

#generate 1D grid
number_of_grid_points = int(NOZZLE_LENGTH_PRIME/GRID_SPACING + 1)
x_grid = np.linspace(0,NOZZLE_LENGTH_PRIME,number_of_grid_points, dtype=FLOAT_PRECISION)

#generate number of steps array
X = np.linspace(0,TIME_STEPS-1,TIME_STEPS,dtype=INT_PRECISION)

#generate initial conditions and arrays
density_prime_t = 1 - 0.3146*x_grid
temperature_prime_t = 1 - 0.2314*x_grid
velocity_prime_t = (0.1 + 1.09*x_grid) \
                     *np.power(temperature_prime_t,0.5) 
area_prime_t = 1 + 2.2*np.power((x_grid-1.5),2) #nozzle shape

#print(velocity_prime_t.dtype)

speed_of_sound = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)
pressure_prime_t = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)

density_gradient_prime_t = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)
velocity_gradient_prime_t = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)
temperature_gradient_prime_t = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)

density_corrected_gradient_prime_t_delta = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)
velocity_corrected_gradient_prime_t_delta = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)
temperature_corrected_gradient_prime_t_delta = np.empty(number_of_grid_points, dtype=FLOAT_PRECISION)

#assign NaNs to all elements to facilitate detection of errors 
speed_of_sound[:] = EMPTY_TYPE
pressure_prime_t[:] = EMPTY_TYPE
density_gradient_prime_t[:] = EMPTY_TYPE
velocity_gradient_prime_t[:] = EMPTY_TYPE
temperature_gradient_prime_t[:] = EMPTY_TYPE
density_corrected_gradient_prime_t_delta[:] = EMPTY_TYPE
velocity_corrected_gradient_prime_t_delta[:] = EMPTY_TYPE
temperature_corrected_gradient_prime_t_delta[:] = EMPTY_TYPE

#print(density_corrected_gradient_prime_t_delta)

#create arrays to store normalised results
density = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
velocity = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
temperature = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
pressure = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
mach_nos = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
density_gradient_average = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)
velocity_gradient_average = np.empty(TIME_STEPS, dtype=FLOAT_PRECISION)

density[:] = EMPTY_TYPE
velocity[:] = EMPTY_TYPE
temperature[:] = EMPTY_TYPE
pressure[:] = EMPTY_TYPE
mach_nos[:] = EMPTY_TYPE
density_gradient_average[:] = EMPTY_TYPE
velocity_gradient_average[:] = EMPTY_TYPE

results = [density, temperature, pressure, mach_nos]

#begin maccormack scheme
for jj in range(TIME_STEPS):

  #predictor step - calculate gradients at internal points 
  for ii in range(1,number_of_grid_points-1):

    density_gradient_prime_t[ii] = \
      -density_prime_t[ii] \
      *((velocity_prime_t[ii+1] - velocity_prime_t[ii])/GRID_SPACING) \
      -density_prime_t[ii]*velocity_prime_t[ii] \
      *((np.log(area_prime_t[ii+1]) - np.log(area_prime_t[ii]))/GRID_SPACING) \
      -velocity_prime_t[ii] \
      *((density_prime_t[ii+1] - density_prime_t[ii])/GRID_SPACING)

    velocity_gradient_prime_t[ii] = \
      -velocity_prime_t[ii] \
      *((velocity_prime_t[ii+1] - velocity_prime_t[ii])/GRID_SPACING) \
      -(1/GAMMA) \
      *( \
        ((temperature_prime_t[ii+1] - temperature_prime_t[ii])/GRID_SPACING) \
        +(temperature_prime_t[ii]/density_prime_t[ii]) \
        *((density_prime_t[ii+1] - density_prime_t[ii])/GRID_SPACING) \
        )

    temperature_gradient_prime_t[ii] = \
      -velocity_prime_t[ii] \
      *((temperature_prime_t[ii+1] - temperature_prime_t[ii])/GRID_SPACING) \
      -(GAMMA-1)*temperature_prime_t[ii] \
      *( \
        ((velocity_prime_t[ii+1] - velocity_prime_t[ii])/GRID_SPACING) \
        +velocity_prime_t[ii] \
        *((np.log(area_prime_t[ii+1]) - np.log(area_prime_t[ii]))/GRID_SPACING) \
        )
  
  #print(temperature_gradient_prime_t)
  
  #time step
  speed_of_sound_prime = np.power(temperature_prime_t,0.5)
  time_steps_prime = \
    COURANT_NUMBER*GRID_SPACING/(speed_of_sound_prime + velocity_prime_t)
  min_time_step_prime = np.min(time_steps_prime)

  #print(time_steps_prime)
  #print(min_time_step_prime)
  
  #predictor - calculate barred quantities at internal points
  for ii in range(1,number_of_grid_points-1):

    density_predicted_prime_t_delta = \
      density_prime_t \
      +density_gradient_prime_t \
      *min_time_step_prime

    velocity_predicted_prime_t_delta = \
      velocity_prime_t \
      +velocity_gradient_prime_t \
      *min_time_step_prime

    temperature_predicted_prime_t_delta = \
      temperature_prime_t \
      +temperature_gradient_prime_t \
      *min_time_step_prime
  
  #print(temperature_predicted_prime_t_delta)

  #assign values to external points of predicted variables
  density_predicted_prime_t_delta[0] = 1
  velocity_predicted_prime_t_delta[0] = velocity_prime_t[0]
  temperature_predicted_prime_t_delta[0] = 1
  
  density_predicted_prime_t_delta[number_of_grid_points-1] = \
    density_prime_t[number_of_grid_points-1]
  velocity_predicted_prime_t_delta[number_of_grid_points-1] = \
    velocity_prime_t[number_of_grid_points-1]
  temperature_predicted_prime_t_delta[number_of_grid_points-1] = \
    temperature_prime_t[number_of_grid_points-1]
  
  #print(velocity_predicted_prime_t_delta)

  #corrector - calculate corrected gradients
  for ii in range(1,number_of_grid_points-1):
    density_corrected_gradient_prime_t_delta[ii] = \
      -density_predicted_prime_t_delta[ii] \
      *( \
        velocity_predicted_prime_t_delta[ii] \
        -velocity_predicted_prime_t_delta[ii-1] \
      ) \
      *(1/GRID_SPACING) \
      -density_predicted_prime_t_delta[ii] \
      *velocity_predicted_prime_t_delta[ii] \
      *( \
        np.log(area_prime_t[ii]) \
        -np.log(area_prime_t[ii-1]) \
      ) \
      *(1/GRID_SPACING) \
      -velocity_predicted_prime_t_delta[ii] \
      *( \
        density_predicted_prime_t_delta[ii] \
        -density_predicted_prime_t_delta[ii-1] \
      ) \
      *(1/GRID_SPACING)

    #print(density_corrected_gradient_prime_t_delta[ii])

    velocity_corrected_gradient_prime_t_delta[ii] = \
      -velocity_predicted_prime_t_delta[ii] \
      *( \
        velocity_predicted_prime_t_delta[ii] \
        -velocity_predicted_prime_t_delta[ii-1] \
        ) \
      *(1/GRID_SPACING) \
      -(1/GAMMA) \
      *( \
        temperature_predicted_prime_t_delta[ii] \
        -temperature_predicted_prime_t_delta[ii-1] \
        ) \
      *(1/GRID_SPACING) \
      -(1/GAMMA) \
      *( \
        temperature_predicted_prime_t_delta[ii] \
        /density_predicted_prime_t_delta[ii]
        ) \
      *( \
        density_predicted_prime_t_delta[ii] \
        -density_predicted_prime_t_delta[ii-1] \
        ) \
      *(1/GRID_SPACING)

    temperature_corrected_gradient_prime_t_delta[ii] = \
      -velocity_predicted_prime_t_delta[ii] \
      *( \
        temperature_predicted_prime_t_delta[ii] \
        -temperature_predicted_prime_t_delta[ii-1] \
        ) \
      *(1/GRID_SPACING) \
      -(GAMMA - 1)*temperature_predicted_prime_t_delta[ii] \
      *( \
        velocity_predicted_prime_t_delta[ii] \
        -velocity_predicted_prime_t_delta[ii-1] \
        ) \
      *(1/GRID_SPACING)\
      -(GAMMA - 1)*temperature_predicted_prime_t_delta[ii] \
      *velocity_predicted_prime_t_delta[ii] \
      *(np.log(area_prime_t[ii]) - np.log(area_prime_t[ii-1])) \
      *(1/GRID_SPACING)

  #print(jj+1,abs(velocity_corrected_gradient_prime_t_delta[15]))

  #calculate average time derivatives
  density_gradient_avg = \
    (density_gradient_prime_t + density_corrected_gradient_prime_t_delta) \
    *0.5
  #print(abs(density_gradient_prime_t[15]))

  velocity_gradient_avg = \
    (velocity_gradient_prime_t + velocity_corrected_gradient_prime_t_delta) \
    *0.5
  
  temperature_gradient_avg = \
    (temperature_gradient_prime_t + temperature_corrected_gradient_prime_t_delta) \
    *0.5

  #print(temperature_gradient_prime_t)
  #print(temperature_corrected_gradient_prime_t_delta)
  #print(temperature_gradient_avg)

  #update variables 
  density_prime_t = density_prime_t + density_gradient_avg*min_time_step_prime
  velocity_prime_t = velocity_prime_t + velocity_gradient_avg*min_time_step_prime
  temperature_prime_t = \
    temperature_prime_t + temperature_gradient_avg*min_time_step_prime
  pressure_prime_t = density_prime_t * temperature_prime_t

  #print(density_prime_t[0])
  #print(temperature_prime_t[0])
  #print(pressure_prime_t[0])
  #print('next')

  #set fixed boundary conditions - needed to avoid nans in update above
  density_prime_t[0] = 1
  temperature_prime_t[0] = 1

  #calculate floating inflow boundary conditions 
  velocity_prime_t[0] = 2*velocity_prime_t[1] - velocity_prime_t[2]
  
  #print(velocity_prime_t[0])

  #calculate floating outflow boundary conditions 
  density_prime_t[number_of_grid_points-1] = \
    2*density_prime_t[number_of_grid_points-2] \
    - density_prime_t[number_of_grid_points-3]

  velocity_prime_t[number_of_grid_points-1] = \
    2*velocity_prime_t[number_of_grid_points-2] \
    - velocity_prime_t[number_of_grid_points-3]

  temperature_prime_t[number_of_grid_points-1] = \
    2*temperature_prime_t[number_of_grid_points-2] \
    -temperature_prime_t[number_of_grid_points-3]
  
  #print(jj)

  #store results
  density[jj] = density_prime_t[15]
  velocity[jj] = velocity_prime_t[15]
  temperature[jj] = temperature_prime_t[15]
  pressure[jj] = pressure_prime_t[15]
  mach_nos[jj] = velocity_prime_t[15]/np.power(temperature_prime_t[15],0.5)
  density_gradient_average[jj] = abs(density_gradient_avg[15])
  velocity_gradient_average[jj] = abs(velocity_gradient_avg[15])

#print(velocity_prime_t.shape)
#print(X.shape)
#print(area_prime_t[15])

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
  axs[kk].scatter(X, results[kk], s=1, color='black')
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
#plt.scatter(X, density_gradient_average, s=1, color='black')
plt.scatter(X, velocity_gradient_average, s=1, color='red')
#plt.scatter(X, temperature_gradient_average, s=1, color='red')
plt.xlabel("Number of Time Steps")
plt.ylabel("Residuals")
plt.yscale('log')

plt.show()

#np.savetxt(r'test1.txt', density_gradient_average, delimiter=",")

#pandas datatable 
df = pd.DataFrame(
    {'x': x_grid.tolist(),
     'area': area_prime_t.tolist(),
     'density': density_prime_t.tolist(),
     'velocity': velocity_prime_t.tolist(),
     'temp': temperature_prime_t.tolist(),
     'pressure':  pressure_prime_t.tolist(),
     'mach': (velocity_prime_t/np.power(temperature_prime_t,0.5)).tolist()
    })

df.index = np.arange(1, len(df) + 1)

#print(df)