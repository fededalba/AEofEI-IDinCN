# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:59:07 2021

@author: Fede
"""

from brian2 import *
import numpy as np
import scipy.io
from scipy import signal

prefs.codegen.target = "numpy"

defaultclock.dt = 0.05*ms

duration = 10000*ms

def single_firingrate(dic):
    firingrates = []
    for i in range(len(dic)):
        firingrates.append(len(dic[i])/(duration / second))
    return(firingrates)

def powerspectrum(signal):
    signal = np.array(signal[20000:])
    Delta_t = duration/(len(signal)*second)
    number_cuts = 40
    data_cuts = np.split(signal, number_cuts)
    asds = []

    for data_cut in data_cuts:
        window = get_window('hann', data_cut.size)
        hwin = data_cut * window
        norm = np.sqrt(2.)/np.sqrt(len(data_cut)) # questa devo capire meglio cos'è
        fftamp = norm * np.fft.rfft(hwin)
        asd = np.abs(fftamp)

    asds.append(asd)
    asd_array = np.asarray(asds)
    asd_frequency = np.linspace(0.0, 1./(2*Delta_t), len(data_cuts[0])//2+1)
    mean_values = np.average(asd_array, axis=0)
    frequencies = np.linspace(0.,1./(2*Delta_t),len(signal)//2+1)
    asd = np.interp(frequencies, asd_frequency, mean_values, 0.,0.)
    psd = asd**2/(window*window).sum()
    return(frequencies, psd)
        

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

tau_I_GABA = 0.5*ms
tau_r_GABA = 0.5*ms
tau_d_GABA = 5*ms

tau_I_AMPA = 1*ms
tau_r_AMPA = 0.4*ms
tau_d_AMPA = 2*ms

g_m_E = 25. * nS
g_m_I = 20. * nS

##definisco le equazioni per i neuroni inibitori
#in questo modello, i neuroni inibitori ricevono sia input eccitatori dal network eccitatorio
#sia input eccitatori da processi poissoniani esterni
eqs_I = '''
dv/dt = (g_m_I*(v-v0)-(I_syn_I+I_poiss_I))/C_m_I : volt (unless refractory)
I_syn_I = g_AMPA_I*(v-Vsyn_E)*s_AMPA + g_GABA_I*(v-Vsyn_I)*s_GABA : ampere
I_poiss_I = g_poiss_I*(v-Vsyn_E)*s_ext_I : ampere

dx_GABA/dt = -x_GABA/tau_r_GABA : 1
ds_GABA/dt = (x_GABA - s_GABA)/tau_d_GABA : 1

dx_AMPA/dt = -x_AMPA/tau_r_AMPA : 1
ds_AMPA/dt = (x_AMPA - s_AMPA)/tau_d_AMPA : 1

dx_ext_I/dt = -x_ext_I/tau_r_AMPA : 1
ds_ext_I/dt = (x_ext_I-s_ext_I)/tau_d_AMPA : 1
'''

##definisco le equazioni per i neuroni eccitatori
#I neuroni eccitatori ricevono gli input inibitori e i processi poissoniani.
eqs_E = '''
dv / dt = (g_m_E*(v-v0)-(I_syn+I_poiss))/C_m_E : volt (unless refractory)
I_syn = g_GABA_E*(v-Vsyn_I)*s : ampere
I_poiss = g_poiss_E*(v-Vsyn_E)*s_ext : ampere

dx/dt = -x/tau_r_GABA : 1
ds/dt = (x - s)/tau_d_GABA : 1

dx_ext/dt = -x_ext/tau_r_AMPA : 1
ds_ext/dt = (x_ext-s_ext)/tau_d_AMPA : 1
'''

##Adesso definisco le popolazioni

G_I = NeuronGroup(N_I, eqs_I, threshold = 'v>theta', reset = 'v = v_reset', refractory = 1*ms, method = 'rk2')
G_I.v = v0 + (v_reset - v0) * rand(len(G_I))
G_E = NeuronGroup(N_E, eqs_E, threshold = 'v>theta', reset = 'v = v_reset', refractory = 2*ms, method = 'rk2')
G_E.v = v0 + (v_reset - v0) * rand(len(G_E))

##Adesso definisco le connessioni
#Dare gli stessi nomi ai pesi potrebbe essere un problema?

#I to E
C_I_E = Synapses(G_I, G_E, 'wie:1', on_pre = 'x += wie')
C_I_E.connect('i!=j', p=0.2)
C_I_E.wie = tau_m_E/tau_r_GABA
C_I_E.delay = tau_I_GABA

#E to I
C_E_I = Synapses(G_E, G_I, 'wei:1', on_pre = 'x_AMPA += wei')
C_E_I.connect('i!=j', p=0.2)
C_E_I.wei = tau_m_I/tau_r_AMPA
C_E_I.delay = tau_I_AMPA

#I to I
C_I_I = Synapses(G_I, G_I, 'wii:1', on_pre = 'x_GABA += wii')
C_I_I.connect('i!=j', p=0.2)
C_I_I.wii = tau_m_I/tau_r_GABA
C_I_I.delay = tau_I_GABA

##Aggiungo gli input poissoniani
P_I = PoissonInput(target=G_I, target_var='x_ext_I', N=800, rate=30*Hz, weight=tau_m_I/tau_r_AMPA)
P_E = PoissonInput(target=G_E, target_var='x_ext', N=800, rate=27.5*Hz, weight=tau_m_E/tau_r_AMPA)


#monitoro il rate di firing della popolazione
r_I = PopulationRateMonitor(G_I)
r_E = PopulationRateMonitor(G_E)

#definisco il raster plot
spikemon_I = SpikeMonitor(G_I)
spikemon_E = SpikeMonitor(G_E)

run(duration)

plt.figure(1)
subplot(1,2,1)
plt.xlim(1000,1100)
plot(r_I.t / ms, r_I.smooth_rate(window = 'gaussian', width = 0.5*ms) / Hz)
#plot(statemon1.t/ms, statemon1.I_poiss[0])
xlabel('Time(ms)')
ylabel('Neuron index');

subplot(1,2,2)
plt.xlim(1000,1100)
plt.ylim(0,20)
plot(spikemon_I.t/ms, spikemon_I.i, '.')
xlabel('Time(ms)')
ylabel('mV');

plt.figure(2)
subplot(1,2,1)
plt.xlim(1000,1100)
plot(r_E.t / ms, r_E.smooth_rate(window = 'gaussian', width = 0.5*ms) / Hz)
#plot(statemon1.t/ms, statemon1.I_poiss[0])
xlabel('Time(ms)')
ylabel('Neuron index');

subplot(1,2,2)
plt.xlim(1000,1100)
plt.ylim(0,20)
plot(spikemon_E.t/ms, spikemon_E.i, '.')
xlabel('Time(ms)')
ylabel('mV');

plt.figure(3)
subplot(1,2,1)
plt.hist(single_firingrate(spikemon_I.spike_trains()), density=True, bins=200)

subplot(1,2,2)
plt.hist(single_firingrate(spikemon_E.spike_trains()), density=True, bins=200)

spectr = powerspectrum(r_I.smooth_rate(window = 'flat', width = 0.5*ms) / Hz)
plt.figure(4)
plt.xlabel('frequency[Hz]')
plt.ylabel('Power spectrum')
plt.xlim(0,300.)
plt.plot(spectr[0], spectr[1])

