# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:53:33 2021

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

    ampamon_onI = StateMonitor(G_I, 'I_AMPA_I', record=list(range(50)))
    gabamon_onI = StateMonitor(G_I, 'I_GABA_I', record=list(range(50)))

    '''ampamon_onE = StateMonitor(G_E, 'I_AMPA_E', record=list(range(100)))
    gabamon_onE = StateMonitor(G_E, 'I_GABA_E', record=list(range(500)))'''

    store()
    ratio_onI = []
    for i in range(num_medie):
        restore()
        run(running_time, report=ProgressBar(), report_period=1*second)

        average_onI = np.mean(ampamon_onI.I_AMPA_I, axis=0)
        average_onE = np.mean(gabamon_onI.I_GABA_I, axis=0)
        ratio = np.mean(average_onI[10000:])/np.mean(average_onE[10000:])
        ratio_onI.append(np.abs(ratio))

    max_point = np.max(ratio_onI)
    min_point = np.min(ratio_onI)
    balance_I = np.mean(ratio_onI)
    #balance_E = np.abs(np.mean(ratio_onItemporalmean))
    print(f'FINISHED {pid}')
    #return balance_I, balance_E
    return balance_I, max_point, min_point



if __name__ == "__main__":
    start_time = time.time()
    num_proc = 8
    #control_parameter = np.linspace(0., 1., 8)
    control_parameter = np.array([0., 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1.])

    clear_cache('cython')

    with multiprocessing.Pool(num_proc) as p:
        results = p.map(running_sim, control_parameter)

    elapsed_time = time.time() - start_time
    print(f'Tempo d esecuzione è {elapsed_time/60} minuti')

    numpy_array = np.array(results)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()

    ratioI, massimi, minimi = transpose_list

    np.savetxt('4kHz_r-Igraph(3)', transpose_list)

    plt.figure(1)
    plt.ylabel('Iampa/Igaba')
    plt.xlabel('r')
    plt.title('4k Hz Input on I')
    plt.scatter(control_parameter, ratioI)
    plt.scatter(control_parameter, massimi, marker='_')
    plt.scatter(control_parameter, minimi, marker='_')

    '''plt.figure(1)
    plt.ylabel('Iampa/Igaba')
    plt.xlabel('r')
    plt.title('4k Hz Input on E')
    plt.scatter(control_parameter, ratioE)'''

    plt.show()