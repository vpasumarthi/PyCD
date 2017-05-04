#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import genfromtxt
from math import pow

ANG2BOHR = 1.8897261264678997
ANG2uM = 1.00e-04
SEC2AUTIME = 4.134137336634339e+16
SEC2NS = 1.00e+09
d = 3

D = []
n_traj = 100
#n_compute = 100 # Since the function subsides early, it is not required to compute for the entire length
for i in range(n_traj):
    time_data = np.loadtxt('Time.dat')[i*101:(i+1)*101] / SEC2AUTIME
    pos_data = np.loadtxt('unwrappedTraj.dat')[i*101:(i+1)*101, :] / ANG2BOHR * ANG2uM
    vel_data = np.diff(pos_data, axis=0) / np.diff(time_data)[:, None]
    n_steps = len(vel_data)
    n_compute = 30 #n_steps - 1; 50 ns  = ~20 steps for 1 electron
    mean = []
    for step_size in range(n_compute):
        vel_sum = 0
        t0 = 0
        n_pairs = n_steps-step_size
        for pair in range(n_pairs):
            t1 = t0 + step_size
            vel_sum += np.sum(np.multiply(vel_data[t0], vel_data[t1]))
            t0 += 1
        mean.append(vel_sum / n_pairs)
    D.append(np.trapz(mean, time_data[1:n_compute+1]) / d)
    
print "Diffusivity obtained from velocity autocorrelation: %4.5f um2/s" % np.mean(D)

'''
plt.figure(1)
plt.axhline(0, color='black')
plt.plot(mean)
plt.xlabel('del_t')
plt.ylabel('<v(o).v(t)>')
plt.title('Velocity Autocorrelation Function')
plt.show()
plt.savefig('Velocity Autocorrelation.jpg')
'''