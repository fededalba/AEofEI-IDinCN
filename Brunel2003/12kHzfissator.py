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
import brian2.numpy_ as np
import logging

logging.basicConfig(level=logging.INFO)

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
r = control_values[2]       #scelgo quale parametro di r studiare

chiamata = 1

def single_firingrate(dic):
    firingrates = []
    for i in range(len(dic)):
        firingrates.append(len(dic[i])/(running_time / second))
    mean_fr = np.mean(firingrates)
    print(f'firing rate medio è {mean_fr}')
    return(firingrates)


def STS(*args):
    '''Funzione che prende i tempi di spikes di più network e ne calcola l'sts.'''
    global chiamata
    #args contiene i tempi di spikes in una tupla generati dalla funzione SpikeMonitor
    if len(args)>1:
        total_spiketimes = []
        for arg in args:
            total_spiketimes.append(arg)
        #metto tutti i tempi di spike in un unica lista
        flat_spiketimes = [item for sublist in total_spiketimes for item in sublist]

    else:
        flat_spiketimes = args[0]
    #genero i bins larghi 1 ms
    bins_width = np.linspace(0, int(running_time/ms), int(running_time/ms))

    plt.figure(chiamata+10)
    plt.title('Istantaneuos firing rate in 1ms bin')
    plt.xlabel('time[ms]')
    plt.ylabel('network activity')
    #hist mi conta le occorrenze in un bin di 1 ms, quindi il numero di spikes in 1 ms
    hist, bin_edges, patches = plt.hist(flat_spiketimes, bins=bins_width)

    #scarto i primi 500 ms
    firing_rates = hist[500:] / second

    #now we compute the autocorrelation
    '''autocorrelation = signal.correlate(firing_rates / Hz, firing_rates / Hz, mode='full')/len(hist)
    plt.figure(15+chiamata)
    plt.title('autocorrelation')
    plt.plot(autocorrelation)'''
    std = np.std(firing_rates / Hz)

    mean_firingrates = np.mean(firing_rates / Hz)

    #print(f'la autocorrelazione alla {chiamata}° è: {autocorrelation[len(autocorrelation)//2]}')
    print(f'il std alla {chiamata}° è: {std}')
    print(f'il firing rate medio alla {chiamata}° è: {mean_firingrates}')
    chiamata += 1
    #return autocorrelation[len(autocorrelation)//2]/mean_firingrates**2
    return (std/mean_firingrates)**2
    
    

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

ampamon_onI = StateMonitor(G_I, 'I_AMPA_I', record=True)
gabamon_onI = StateMonitor(G_I, 'I_GABA_I', record=True)


run(running_time, report=ProgressBar(), report_period=1*second)


Iampa_neuronmean_onI = np.mean(ampamon_onI.I_AMPA_I / amp, axis=0)
Igaba_neuronmean_onI = np.mean(gabamon_onI.I_GABA_I / amp, axis=0)

Iampa_temporalmean_onI = np.mean(Iampa_neuronmean_onI[10000:])
Igaba_temporalmean_onI = np.mean(Igaba_neuronmean_onI[10000:])

ratio = Iampa_temporalmean_onI/Igaba_temporalmean_onI

print(f'il rapport ampa gaba è {ratio}')

elapsed_time = time.time() - start_time

pop_osc_I = r_I.smooth_rate(window = 'gaussian', width = 0.5*ms)[10000:] / Hz
pop_osc_E = r_E.smooth_rate(window = 'gaussian', width = 0.5*ms)[10000:] / Hz
pop_osc_tot = pop_osc_I + pop_osc_E

plt.figure(1)
#plt.xlim(1000,1200)
plt.title('population oscillation')
plot(r_I.t[10000:] / ms, pop_osc_I)
plot(r_E.t[10000:] / ms, pop_osc_E)
xlabel('Time(ms)')
ylabel('Population frequency');

plt.figure(2)
subplot(1,2,1)
#plt.xlim(1000,1200)
#plt.ylim(0,20)
plt.title('Rasterplot neuroni inibitori')
plot(M_I.t/ms, M_I.i, '.')
ylabel('Neuron index');

subplot(1,2,2)
plt.title('Rasterplot neuroni eccitatori')
#xlim(1000,1200)
#ylim(0,20)
plot(M_E.t/ms, M_E.i, '.')
ylabel('Neuron index');

plt.figure(3)
subplot(1,2,1)
title('firing rate network inibitorio')
hist(single_firingrate(M_I.spike_trains()), density=True, bins=50)
xlabel('Frequency[Hz]')
ylabel('Fraction of cell');

subplot(1,2,2)
title('firing rate network eccitatorio')
hist(single_firingrate(M_E.spike_trains()), density=True, bins=50)
xlabel('Frequency[Hz]');


freq, spectr_tot = signal.welch(pop_osc_tot, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc_tot)/n_split))
sts_tot = STS(M_I.t/ms, M_E.t/ms)

freq, spectr_I = signal.welch(pop_osc_I, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc_I)/n_split))
sts_I = STS(M_I.t/ms)

freq, spectr_E = signal.welch(pop_osc_E, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc_E)/n_split))
sts_E = STS(M_E.t/ms)

print(f'sts network: tot={sts_tot}, inib={sts_I}, exc={sts_E}')


plt.figure(4)
title('total power spectr')
xlabel('frequency[Hz]')
ylabel('Power spectrum')
#plt.xlim(0,300.)
plot(freq, spectr_tot);

plt.figure(5)
subplot(1,2,1)
title('inhibitory power spectr')
xlabel('frequency[Hz]')
ylabel('Power spectrum')
#plt.xlim(0,300.)
plot(freq, spectr_I);

subplot(1,2,2)
title('excitatory power spectr')
xlabel('frequency[Hz]')
#plt.xlim(0,300.)
plot(freq, spectr_E);

plt.figure(6)
plot(pop_osc_tot)
title('total network oscillation')
xlabel('Time(ms)')
ylabel('Population frequency');
