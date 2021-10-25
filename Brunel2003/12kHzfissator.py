# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:20:32 2021

@author: Fede
"""

import os
import multiprocessing
import time
from scipy import signal
from brian2 import *

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

#%%

defaultclock.dt = 0.05*ms

running_time = 10000*ms

step_temporale= 0.00005

n_split = 20        #numero di split con cui dividere i dati per applicare il welch method
num_medie = 10

control_values = np.linspace(0., 1., 8)
r = control_values[6]       #scelgo quale parametro di r studiare

def single_firingrate(n, *args):
    '''n è il numero di dizionari che vengono passati '''
    fr_names = ['fr%d'%k for k in range(n)]
    fr = {}
    s = 0
    for dic in args:
        fr[fr_names[s]] = []
        for i in range(len(dic)):
            fr[fr_names[s]].append(len(dic[i])/(running_time / second))
        s += 1
    return(fr)

def STS(data, n, *args):
    ''' n numero di dizionari che vengono passati'''
    data_cuts = np.split(data, 5)
    autocorr_splits = []
    for data_cut in data_cuts:
        autocorr_splits.append(signal.correlate(data_cut, data_cut, mode='full')/len(data_cut))
    autocorr_mean = np.mean(autocorr_splits, axis=0)
    dics_names = ['fr%d'%k for k in range(n)]
    dics = {}
    for i in range(len(args)):
        dics[dics_names[i]] = args[i]
##cerca di capire come farlo
    fr = single_firingrate(n, fr[])
    print(fr)
    fr_names = ['fr%d'%k for k in range(n)]
    firing_rates = fr['fr0']
    for i in range(1,n):
        firing_rates.append(fr[fr_names[i]])


    fr_averaged = np.mean(firing_rates)
    return autocorr_mean[len(autocorr_mean)//2]/fr_averaged**2

start_scope()
start_time = time.time()
##tensioni e popolazioni
N_I = 1000
N_E = 4000
v0 = -70*mV #tensione di riposo
theta = -52*mV
v_reset = -59*mV
Vsyn_I = -70*mV

##Neuroni inibitori
tau_m_I = 10*ms
g_GABA_E = 2.5*nS #conduttanza recettori inibitori su neuroni eccitatori
g_GABA_I = 4*nS #conduttanza recettori inibitori su neuroni ini
g_poiss_I = 0.4*nS #conduttanza per sinapsi esterne
Vsyn_I = -70*mV #reversal tension sinapsi inibitorie
C_m_I = 0.2*nF
g_m_I = 20.*nS

##Neuroni eccitatori
tau_m_E = 20*ms
C_m_E = 0.5*nF
g_AMPA_E = 0.19*nS
g_AMPA_I = 0.3*nS   #conduttanza recettori eccitatori su neuroni inibitori
g_poiss_E = 0.25*nS
Vsyn_E = 0*mV
g_m_E = 25. * nS

#GABA
tau_I_GABA = 1*ms
tau_r_GABA = 0.5*ms
tau_d_GABA = 5*ms

#AMPA
tau_I_AMPA = 1*ms
tau_r_AMPA = 0.2*ms
tau_d_AMPA = 2*ms

# EXTERNAL INPUT
external_rate_on_EXC = 5 * Hz
external_rate_on_INH = 5 * Hz


eqs_I = '''
dv / dt = (g_m_I*(v-v0)-(I_syn+I_poiss))/C_m_I : volt (unless refractory)
I_syn = I_AMPA_I + I_GABA_I : ampere
I_poiss = g_poiss_I*v*s_ext : ampere

I_GABA_I = g_GABA_I*(v-Vsyn_I)*sI_GABA : ampere
dxI_GABA/dt = -xI_GABA/tau_r_GABA : 1
dsI_GABA/dt = (xI_GABA - sI_GABA)/tau_d_GABA : 1

I_AMPA_I = g_AMPA_I*r*(v-Vsyn_E)*sI_AMPA : ampere
dxI_AMPA/dt = -xI_AMPA/tau_r_AMPA : 1
dsI_AMPA/dt = (xI_AMPA - sI_AMPA)/tau_d_AMPA : 1

dx_ext/dt = -x_ext/tau_r_AMPA : 1
ds_ext/dt = (x_ext-s_ext)/tau_d_AMPA : 1
'''

eqs_E = '''
dv/dt = (g_m_E*(v-v0)-(I_syn+I_poiss))/C_m_E : volt (unless refractory)
I_syn = I_GABA_E + I_AMPA_E : ampere
I_poiss = g_poiss_E*v*s_ext : ampere

I_GABA_E = g_GABA_E*(v-Vsyn_I)*sE_GABA : ampere
dxE_GABA/dt = -xE_GABA/tau_r_GABA : 1
dsE_GABA/dt = (xE_GABA - sE_GABA)/tau_d_GABA : 1

I_AMPA_E = g_AMPA_E*r*(v-Vsyn_E)*sE_AMPA : ampere
dxE_AMPA/dt = -xE_AMPA/tau_r_AMPA : 1
dsE_AMPA/dt = (xE_AMPA - sE_AMPA)/tau_d_AMPA : 1

dx_ext/dt = -x_ext/tau_r_AMPA : 1
ds_ext/dt = (x_ext-s_ext)/tau_d_AMPA : 1
'''


G_I = NeuronGroup(N_I, eqs_I, threshold = 'v>theta', reset = 'v = v_reset', refractory = 1*ms, method = 'rk2')
G_I.v = v0 + (v_reset - v0) * rand(len(G_I))
G_E = NeuronGroup(N_E, eqs_E, threshold = 'v>theta', reset = 'v = v_reset', refractory = 2*ms, method = 'rk2')
G_E.v = v0 + (v_reset - v0) * rand(len(G_E))

Wie = tau_m_E/tau_r_GABA
C_I_E = Synapses(G_I, G_E, on_pre = 'xE_GABA += Wie')
C_I_E.connect(p=0.2)
C_I_E.delay = tau_I_GABA

#E to I
Wei = tau_m_I/tau_r_AMPA
C_E_I = Synapses(G_E, G_I, on_pre = 'xI_AMPA += Wei')
C_E_I.connect(p=0.2)
C_E_I.delay = tau_I_AMPA

#I to I
Wii = tau_m_I/tau_r_GABA
C_I_I = Synapses(G_I, G_I, on_pre = 'xI_GABA += Wii')
C_I_I.connect(p=0.2)
C_I_I.delay = tau_I_GABA

#E to E
Wee = tau_m_E/tau_r_AMPA
C_E_E = Synapses(G_E, G_E, on_pre = 'xE_AMPA += Wee')
C_E_E.connect(p=0.2)
C_E_E.delay = tau_I_AMPA


WextE = (tau_m_E)/tau_r_AMPA
WextI = (tau_m_I)/tau_r_AMPA


if (external_rate_on_INH / Hz) >= 10 or (external_rate_on_EXC / Hz) >= 10:
    external_rate_E = Equations('''
                                rate_0 = external_rate_on_EXC*200 : Hz
                                ''')

    external_rate_I = Equations('''
                                rate_1 = external_rate_on_INH*200 : Hz
                                ''')
    
    poisson_group_onI = ['p_I%d'%i for i in range(4)]
    poisson_group_onE = ['p_E%d'%i for i in range(4)]
    for k in range(4):
        globals()[poisson_group_onI[k]] = NeuronGroup(N_I, model=external_rate_I, threshold='rand()<rate_1*dt')
        globals()[poisson_group_onE[k]] = NeuronGroup(N_E, model=external_rate_E, threshold='rand()<rate_0*dt')

    synapsesonI_names = ['s_I%d'%i for i in range(4)]
    synapsesonE_names = ['s_E%d'%i for i in range(4)]
    for k in range(4):
        globals()[synapsesonE_names[k]] = Synapses(globals()[poisson_group_onE[k]], G_E, on_pre='x_ext += WextE', delay=tau_I_AMPA)
        globals()[synapsesonE_names[k]].connect('i==j')

        globals()[synapsesonI_names[k]] = Synapses(globals()[poisson_group_onI[k]], G_I, on_pre='x_ext += WextI', delay=tau_I_AMPA)
        globals()[synapsesonI_names[k]].connect('i==j')

else:
    external_rate_E = Equations('''
                                rate_0 = external_rate_on_EXC*800 : Hz
                                ''')

    external_rate_I = Equations('''
                                rate_1 = external_rate_on_INH*800 : Hz
                                ''')

    input_on_Exc = NeuronGroup(N_E, model=external_rate_E, threshold='rand()<rate_0*dt')
    input_on_Inh = NeuronGroup(N_I, model=external_rate_I, threshold='rand()<rate_1*dt')

    C_P_E = Synapses(input_on_Exc, G_E, on_pre='x_ext += WextE', delay=tau_I_AMPA)
    C_P_E.connect('i==j')

    C_P_I = Synapses(input_on_Inh, G_I, on_pre='x_ext += WextI', delay=tau_I_AMPA)
    C_P_I.connect('i==j')


#monitoro il rate di firing della popolazione
r_I = PopulationRateMonitor(G_I)
r_E = PopulationRateMonitor(G_E)
M_I = SpikeMonitor(G_I)
M_E = SpikeMonitor(G_E)

run(running_time, report=ProgressBar(), report_period=1*second)

elapsed_time = time.time() - start_time

pop_osc = r_I.smooth_rate(window = 'gaussian', width = 1*ms)[5000:] / Hz
pop_osc_E = r_E.smooth_rate(window = 'gaussian', width = 1*ms)[5000:] / Hz
pop_osc_tot = pop_osc + pop_osc_E

plt.figure(1)
plt.xlim(1000,1200)
plot(r_I.t[5000:] / ms, pop_osc)
plot(r_E.t[5000:] / ms, pop_osc_E)
#plot(statemon1.t/ms, statemon1.I_poiss[0])
xlabel('Time(ms)')
ylabel('Population frequency');

plt.figure(2)
subplot(1,2,1)
plt.xlim(1000,1200)
plt.ylim(0,20)
plot(M_I.t/ms, M_I.i, '.')
ylabel('Neuron index');

subplot(1,2,2)
plt.xlim(1000,1200)
plt.ylim(0,20)
plot(M_E.t/ms, M_E.i, '.')
ylabel('Neuron index');

#i dizionari finali avranno nomi tipo fr0, fr1 etc etc
firing_rates = single_firingrate(2, M_I.spike_trains(), M_E.spike_trains())

plt.figure(3)
subplot(1,2,1)
plt.hist(firing_rates['fr0'], density=True, bins=50)
xlabel('Frequency[Hz]')
ylabel('Fraction of cell');

subplot(1,2,2)
plt.hist(firing_rates['fr1'], density=True, bins=50)
xlabel('Frequency[Hz]');


freq, spectr = signal.welch(pop_osc_tot, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc)/n_split))
sts_tot = STS(pop_osc_tot, 2, M_I.spike_trains(), M_E.spike_trains())


plt.figure(4)
plt.xlabel('frequency[Hz]')
plt.ylabel('Power spectrum')
#plt.xlim(0,300.)
plt.plot(freq, spectr)
