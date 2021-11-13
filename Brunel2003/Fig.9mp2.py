# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 18:49:50 2021

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

def STS(*args):
    '''Funzione che prende i tempi di spikes di più network e ne calcola l'sts.'''
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

    #hist mi conta le occorrenze in un bin di 1 ms, quindi il numero di spikes in 1 ms
    hist, bin_edges = np.histogram(flat_spiketimes, bins=bins_width)

    #scarto i primi 500 ms
    firing_rates = hist[500:] / second

    #now we compute the autocovariance in 0
    std = np.std(firing_rates / Hz)

    mean_firingrates = np.mean(firing_rates / Hz)

    return (std/mean_firingrates)**2
    

defaultclock.dt = 0.05*ms

running_time = 10000*ms

step_temporale= 0.00005

n_split = 20        #numero di split con cui dividere i dati per applicare il welch method
num_medie = 5

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

def running_sim(r):
    pid = os.getpid()
    print(f'RUNNING {pid}')

    ##definisco le equazioni per i neuroni inibitori
    #in questo modello, i neuroni inibitori ricevono sia input eccitatori dal network eccitatorio
    #sia input eccitatori da processi poissoniani esterni
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

    ##definisco le equazioni per i neuroni eccitatori
    #I neuroni eccitatori ricevono gli input inibitori e i processi poissoniani.
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

    ##Adesso definisco le popolazioni

    G_I = NeuronGroup(N_I, eqs_I, threshold = 'v>theta', reset = 'v = v_reset', refractory = 1*ms, method = 'rk2')
    G_I.v = v0 + (v_reset - v0) * rand(len(G_I))
    G_E = NeuronGroup(N_E, eqs_E, threshold = 'v>theta', reset = 'v = v_reset', refractory = 2*ms, method = 'rk2')
    G_E.v = v0 + (v_reset - v0) * rand(len(G_E))

    ##Adesso definisco le connessioni
    #Dare gli stessi nomi ai pesi potrebbe essere un problema?

    #I to E
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

    WextI = (tau_m_I)/tau_r_AMPA
    WextE = (tau_m_E)/tau_r_AMPA

    if (external_rate_on_INH / Hz) >= 10 or (external_rate_on_EXC / Hz) >= 10:
        external_rate_E = Equations('''
                                    rate_0 = external_rate_on_EXC*200 : Hz
                                    ''')
        external_rate_I = Equations('''
                                    rate_1 = external_rate_on_INH*200 : Hz
                                    ''')

        poisson_group_onI = [f'p_I%d_{pid}'%i for i in range(4)]
        poisson_group_onE = [f'p_E%d_{pid}'%i for i in range(4)]
        for k in range(4):
            globals()[poisson_group_onI[k]] = NeuronGroup(N_I, model=external_rate_I, threshold='rand()<rate_1*dt')
            globals()[poisson_group_onE[k]] = NeuronGroup(N_E, model=external_rate_E, threshold='rand()<rate_0*dt')

        synapsesonI_names = [f's_I%d_{pid}'%i for i in range(4)]
        synapsesonE_names = [f's_E%d_{pid}'%i for i in range(4)]
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

        #collego lo spikegenerator ai neuroni.
        C_P_E = Synapses(input_on_Exc, G_E, on_pre='x_ext += WextE', delay=tau_I_AMPA)
        C_P_E.connect('i==j')

        C_P_I = Synapses(input_on_Inh, G_I, on_pre='x_ext += WextI', delay=tau_I_AMPA)
        C_P_I.connect('i==j')


    #monitoro il rate di firing della popolazione
    r_I = PopulationRateMonitor(G_I)
    M_I = SpikeMonitor(G_I)
    M_E = SpikeMonitor(G_E)

    ampamon_onI = StateMonitor(G_I, 'I_AMPA_I', record=list(range(0,99)))
    gabamon_onI = StateMonitor(G_I, 'I_GABA_I', record=list(range(0,99)))

    store()
    max_psd_freq = []
    sts = []
    ratio = []
    for i in range(num_medie):
        restore()
        run(running_time, report=ProgressBar(), report_period=1*second)

        pop_osc = r_I.smooth_rate(window = 'gaussian', width = 1*ms)[20000:] / Hz

        freq, spectr = signal.welch(pop_osc, fs=1./step_temporale, window='hann', nperseg=int(len(pop_osc)/n_split))
        freq_max_index = np.argmax(spectr)
        max_psd_freq.append(freq[freq_max_index])

        sts.append(STS(M_I.t/ms, M_E.t/ms))

        Iampa_neuronmean_onI = np.mean(ampamon_onI.I_AMPA_I / amp, axis=0)
        Igaba_neuronmean_onI = np.mean(gabamon_onI.I_GABA_I / amp, axis=0)
        Iampa_temporalmean_onI = np.mean(Iampa_neuronmean_onI[10000:])
        Igaba_temporalmean_onI = np.mean(Igaba_neuronmean_onI[10000:])
        ratio.append(np.abs(Iampa_temporalmean_onI/Igaba_temporalmean_onI))
    
    sts_mean = np.mean(sts)
    balance = np.mean(ratio)
    max_psd_freq_mean = np.mean(max_psd_freq)
    res = (max_psd_freq_mean,sts_mean, balance)
    print(f'FINISHED {pid}')
    return max_psd_freq_mean,sts_mean, balance



if __name__ == "__main__":
    start_time = time.time()
    num_proc = 8
    control_parameter = np.linspace(0.85, 0.95, 8)

    #clear_cache('cython')

    with multiprocessing.Pool(num_proc) as p:
        results = p.map(running_sim, control_parameter)

    elapsed_time = time.time() - start_time
    print(f'Tempo d esecuzione è {elapsed_time/60} minuti')

    numpy_array = np.array(results)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()

    max_frequencies, sts, ratio_currents = transpose_list

    np.savetxt('6kHzInput_resultsfocused[0.85-0.95]', transpose_list)

    plt.figure(1)
    plt.xlabel('Iampa/Igaba')
    plt.ylabel('Frequency population')
    plt.title('6k Hz Input')
    plt.scatter(ratio_currents, max_frequencies, c=sts, cmap='plasma')
    plt.colorbar()

    plt.show()
