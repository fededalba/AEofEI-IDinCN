# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:01:46 2021

@author: Fede
"""
#%%
from brian2 import *
import numpy as np
import scipy.io
from scipy import signal
import time

class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0
    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of˓→line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")

prefs.codegen.target = "numpy"

defaultclock.dt = 0.05*ms
step_temporale= 0.00005

n_split = 20        #numero di split con cui dividere i dati per applicare il welch method

duration = 10000*ms

def plot_func(M):
    train_spike = M.spike_trains()
    plt.hist(single_firingrate(train_spike), density=True, bins=50)
    xlabel('Frequency[Hz]')
    ylabel('Fraction of cell')
    

def single_firingrate(dic):
    firingrates = []
    for i in range(len(dic)):
        firingrates.append(len(dic[i])/(duration / second))
    return(firingrates)

def STS(data, dic):
    data_cuts = np.split(data, 5)
    autocorr_splits = []
    for data_cut in data_cuts:
        autocorr_splits.append(signal.correlate(data_cut, data_cut, mode='full')/len(data_cut))
    autocorr_mean = np.mean(autocorr_splits, axis=0)
    fr = single_firingrate(dic)
    fr_averaged = np.mean(fr)
    return autocorr_mean[len(autocorr_mean)//2]/fr_averaged**2

start_scope()
start_time = time.time()

N = 1000
tau_m = 10*ms
v0 = -70*mV #tensione di riposo
theta = -52*mV
v_reset = -59*mV
g_GABA_I = 4*nS #conduttanza recettori inibitori su neuroni ini
g_poiss_I = 0.4*nS #conduttanza per sinapsi esterne
V_syn_I = -70*mV #reversal tension sinapsi inibitorie
C_m = 0.2*nF
g_m_I = 20.*nS

tau_I_GABA = 1*ms
tau_r_GABA = 0.5*ms
tau_d_GABA = 5*ms

tau_I_AMPA = 1*ms
tau_r_AMPA = 0.5*ms
tau_d_AMPA = 2*ms

#dv/dt = (v0 - v)/tau_m - (I_syn + I_poiss)/C_m : volt (unless refractory)

eqs = '''
dv / dt = (g_m_I*(v-v0)-(I_syn+I_poiss))/C_m : volt (unless refractory)
I_syn = g_GABA_I*(v-V_syn_I)*s : ampere
I_poiss = g_poiss_I*v*s_ext : ampere

dx/dt = -x/tau_r_GABA : 1
ds/dt = (x - s)/tau_d_GABA : 1

dx_ext/dt = -x_ext/tau_r_AMPA : 1
ds_ext/dt = (x_ext-s_ext)/tau_d_AMPA : 1
'''

G = NeuronGroup(N, eqs, threshold = 'v>theta', reset = 'v = v_reset', refractory = 1*ms, method = 'rk2')
G.v = [v_reset]*N

Wrec = tau_m/tau_r_GABA
S = Synapses(G, G, on_pre = 'x += Wrec', delay = tau_I_GABA)
S.connect(condition = 'i!=j', p=0.2)

#%%

#P = PoissonInput(target=G, target_var='x_ext', N=800, rate=15*Hz, weight=tau_m/tau_r_AMPA)

external_rate_on_INH = 15 * Hz
external_rate_I = Equations('''
rate_1 = external_rate_on_INH*200 : Hz
''')

#input_on_Inh = NeuronGroup(N, model=external_rate_I, threshold='rand()<rate_1*dt')

Wext = tau_m/tau_r_AMPA
#C_P_I = Synapses(input_on_Inh, G, on_pre='x_ext += Wext', delay=tau_I_AMPA)
#C_P_I.connect('i==j')

poisson_groups = ['p%d'%i for i in range(4)]
for name in poisson_groups:
    globals()[name] = NeuronGroup(N, model=external_rate_I, threshold='rand()<rate_1*dt')

k = 0
synapses_names = ['s%d'%i for i in range(4)]
for name in synapses_names:
    globals()[name] = Synapses(globals()['p%d'%k], G, on_pre='x_ext+=Wext', delay=tau_I_AMPA)
    globals()[name].connect('i==j')
    k += 1

#print(collect())

spikemon = SpikeMonitor(G)

r_I = PopulationRateMonitor(G)

run(duration, report=ProgressBar(), report_period=1*second)

elapsed_time = time.time() - start_time
pop_osc = r_I.smooth_rate(window = 'gaussian', width = 1*ms)[20000:] / Hz

plt.figure(1)
subplot(1,2,1)
plt.xlim(1000,1200)
plot(r_I.t[20000:] / ms, pop_osc)
#plot(statemon1.t/ms, statemon1.I_poiss[0])
xlabel('Time(ms)')
ylabel('Population frequency');

subplot(1,2,2)
plt.xlim(1000,1200)
plt.ylim(0,20)
plot(spikemon.t/ms, spikemon.i, '.')
ylabel('Neuron index');

plt.figure(3)
plt.hist(single_firingrate(spikemon.spike_trains()), density=True, bins=50)
xlabel('Frequency[Hz]')
ylabel('Fraction of cell');

#freq, spectr = powerspectrum(pop_osc)
freq, spectr = signal.welch(pop_osc, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc)/n_split))
sts = STS(pop_osc, spikemon.spike_trains())

plt.figure(4)
plt.xlabel('frequency[Hz]')
plt.ylabel('Power spectrum')
#plt.xlim(0,300.)
plt.plot(np.array(freq), spectr)


