# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 15:49:53 2021

@author: Fede
"""

from brian2 import *
import numpy as np
import scipy.io
from scipy import signal

prefs.codegen.target = "numpy"

defaultclock.dt = 0.05*ms

running_time =10000*ms

step_temporale= 0.00005

n_split = 20        #numero di split con cui dividere i dati per applicare il welch method

def single_firingrate(dic):
    firingrates = []
    for i in range(len(dic)):
        firingrates.append(len(dic[i])/(running_time / second))
    print(firingrates)
    return(firingrates)


start_scope()
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

##Neuroni eccitatori
tau_m_E = 20*ms
C_m_E = 0.5*nF
g_AMPA_E = 0.19*nS  #conduttanza recettori eccitatori su neuroni eccitatori
g_AMPA_I = 0.3*nS   #conduttanza recettori eccitatori su neuroni inibitori
g_poiss_E = 0.25*nS
Vsyn_E = 0*mV

##dinamiche temporali
#le connessioni I->I e I->E hanno le stesse dinamiche temporali.


#fig6
#ricordati di cambiare anche il tempo nei plot

#I->I
tau_I_II = 0.5*ms
tau_r_II = 0.5*ms
tau_d_II = 5*ms

#E->E
tau_I_EE = 0.5*ms
tau_r_EE = 0.4*ms
tau_d_EE = 2*ms

#E->I
tau_I_EI = 0.5*ms
tau_r_EI = 0.2*ms
tau_d_EI = 1*ms
'''
#fig 7

#I->I
tau_I_II = 1.5*ms
tau_r_II = 1.5*ms
tau_d_II = 8*ms

#E->E
tau_I_EE = 1.5*ms
tau_r_EE = 0.4*ms
tau_d_EE = 2*ms

#E->I
tau_I_EI = 1.5*ms
tau_r_EI = 0.2*ms
tau_d_EI = 1*ms
'''
##definisco le equazioni per i neuroni inibitori
#in questo modello, i neuroni inibitori ricevono sia input eccitatori dal network eccitatorio
#sia input eccitatori da processi poissoniani esterni
eqs_I = '''
dv/dt = (v0 - v)/tau_m_I - (I_syn + I_poiss)/C_m_I : volt (unless refractory)
I_syn = I_AMPA + I_GABA : ampere
I_poiss = g_poiss_I*v*s_ext : ampere

I_GABA = g_GABA_I*(v-Vsyn_I)*sI_GABA : ampere
dxI_GABA/dt = -xI_GABA/tau_r_II : 1
dsI_GABA/dt = (xI_GABA - sI_GABA)/tau_d_II : 1

I_AMPA = g_AMPA_I*(v-Vsyn_E)*sI_AMPA : ampere
dxI_AMPA/dt = -xI_AMPA/tau_r_EI : 1
dsI_AMPA/dt = (xI_AMPA - sI_AMPA)/tau_d_EI : 1

dx_ext/dt = -x_ext/tau_r_EI : 1
ds_ext/dt = (x_ext-s_ext)/tau_d_EI : 1
'''

##definisco le equazioni per i neuroni eccitatori
#I neuroni eccitatori ricevono gli input inibitori e i processi poissoniani.
eqs_E = '''
dv/dt = (v0 - v)/tau_m_E - (I_syn + I_poiss)/C_m_E : volt (unless refractory)
I_syn = I_GABA + I_AMPA : ampere
I_poiss = g_poiss_E*v*s_ext : ampere

I_GABA = g_GABA_E*(v-Vsyn_I)*sE_GABA : ampere
dxE_GABA/dt = -xE_GABA/tau_r_II : 1
dsE_GABA/dt = (xE_GABA - sE_GABA)/tau_d_II : 1

I_AMPA = g_AMPA_E*(v-Vsyn_E)*sE_AMPA : ampere
dxE_AMPA/dt = -xE_AMPA/tau_r_EE : 1
dsE_AMPA/dt = (xE_AMPA - sE_AMPA)/tau_d_EE : 1

dx_ext/dt = -x_ext/tau_r_EE : 1
ds_ext/dt = (x_ext-s_ext)/tau_d_EE : 1
'''

##Adesso definisco le popolazioni

G_I = NeuronGroup(N_I, eqs_I, threshold = 'v>theta', reset = 'v = v_reset', refractory = 1*ms, method = 'rk2')
G_I.v = v0 + (v_reset - v0) * rand(len(G_I))
G_E = NeuronGroup(N_E, eqs_E, threshold = 'v>theta', reset = 'v = v_reset', refractory = 2*ms, method = 'rk2')
G_E.v = v0 + (v_reset - v0) * rand(len(G_E))

##Adesso definisco le connessioni
#Dare gli stessi nomi ai pesi potrebbe essere un problema?

#I to E
Wie = tau_m_E/tau_r_II
C_I_E = Synapses(G_I, G_E, on_pre = 'xE_GABA += Wie')
C_I_E.connect(p=0.2)
C_I_E.delay = tau_I_II

#E to I
Wei = tau_m_I/tau_r_EI
C_E_I = Synapses(G_E, G_I, on_pre = 'xI_AMPA += Wei')
C_E_I.connect(p=0.2)
C_E_I.delay = tau_I_EI

#I to I
Wii = tau_m_I/tau_r_II
C_I_I = Synapses(G_I, G_I, on_pre = 'xI_GABA += Wii')
C_I_I.connect(p=0.2)
C_I_I.delay = tau_I_II

#E to E
Wee = tau_m_E/tau_r_EE
C_E_E = Synapses(G_E, G_E, on_pre = 'xE_AMPA += Wee')
C_E_E.connect(p=0.2)
C_E_E.delay = tau_I_EE


# EXTERNAL INPUT
external_rate_on_EXC = 5 * Hz
external_rate_on_INH = 5 * Hz

external_rate_E = Equations('''
rate_0 = external_rate_on_EXC*800 : Hz
''')
external_rate_I = Equations('''
rate_1 = external_rate_on_INH*800 : Hz
''')

input_on_Exc = NeuronGroup(N_E, model=external_rate_E, threshold='rand()<rate_0*dt')
input_on_Inh = NeuronGroup(N_I, model=external_rate_I, threshold='rand()<rate_1*dt')


WextE = (tau_m_E)/tau_r_EE
C_P_E = Synapses(input_on_Exc, G_E, on_pre='x_ext += WextE', delay=tau_I_EE)
C_P_E.connect('i==j')

WextI = (tau_m_I)/tau_r_EI
C_P_I = Synapses(input_on_Inh, G_I, on_pre='x_ext += WextI', delay=tau_I_EI)
C_P_I.connect('i==j')

#monitoro il rate di firing della popolazione
r_I = PopulationRateMonitor(G_I)
r_E = PopulationRateMonitor(G_E)

#definisco il raster plot
spikemon_I = SpikeMonitor(G_I)
spikemon_E = SpikeMonitor(G_E)

run(running_time)

pop_osc = r_I.smooth_rate(window = 'gaussian', width = 0.5*ms)[20000:] / Hz

plt.figure(1)
plt.xlim(1000,1200)
plot(r_I.t[20000:] / ms, pop_osc)
plot(r_E.t / ms, r_E.smooth_rate(window = 'gaussian', width = 0.5*ms) / Hz)
#plot(statemon1.t/ms, statemon1.I_poiss[0])
xlabel('Time(ms)')
ylabel('Neuron index');

plt.figure(2)
subplot(1,2,1)
plt.xlim(1000,1200)
plt.ylim(0,20)
plot(spikemon_I.t/ms, spikemon_I.i, '.')
#plot(statemon1.t/ms, statemon1.I_poiss[0])
xlabel('Time(ms)')
ylabel('Neuron index');

subplot(1,2,2)
plt.xlim(1000,1200)
plt.ylim(0,20)
plot(spikemon_E.t/ms, spikemon_E.i, '.')
xlabel('Time(ms)')
ylabel('mV');

plt.figure(3)
subplot(1,2,1)
plt.hist(single_firingrate(spikemon_I.spike_trains()), density=True, bins=30)

subplot(1,2,2)
plt.hist(single_firingrate(spikemon_E.spike_trains()), density=True, bins=10)

freq, spectr = signal.welch(pop_osc, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc)/n_split))

plt.figure(4)
plt.xlabel('frequency[Hz]')
plt.ylabel('Power spectrum')
#plt.xlim(0,300.)
plt.plot(np.array(freq), spectr)
