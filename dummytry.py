# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:11:38 2021

@author: Fede
"""

from brian2 import *
import brian2.numpy_ as np
import time

defaultclock.dt = 0.05*ms
tau = 1*ms
running_time = 100*ms



start_scope()
start_time = time.time()
eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(10, eqs, threshold='v > 0.8', reset='v = 0', method='rk2')
G.v = 'rand()'

spikemon = SpikeMonitor(G)
run(running_time)
#print(list(spikemon.t/ms))

bins_width = np.linspace(0., 100, 100)

#plt.figure(1)
#hist, bin_edges, patches = plt.hist(list(spikemon.t/ms), bins=bins_width)
hist, bin_edges = np.histogram(list(spikemon.t/ms), bins=bins_width)
print(hist)


elapsed_time = time.time() - start_time

plt.figure(2)
plot(spikemon.t/ms, spikemon.i, '.')
