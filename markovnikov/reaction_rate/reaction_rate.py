from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

h = 6.626070040*10**(-34) # J * s
k = 1.38064852 * 10**(-23) # J / K
R = 8.314 # J / mol / K
da = 1.660 * 10**(-27) # kg
avogadro = 6.022 * 10**(23) # mol**(-1)
hatoj = 2625.5 * 1000 # J /mol
c = 2.99792458 * 10**10 # cm / s

alu = 5.291 * 10**(-11) # alu to m
amu = 1822.0 * 9.109 * 10**(-31) # amu to kg

m_c3h6 = 43 * da
m_hf = 20 * da
m_c3h7f = 63 * da

class QParser(object):
    def __init__(self, filename, m, temperature, linear = False, qlb = 7):
        self.filename = filename
        self.m = m
        self.temperature = temperature

        self.data = self.read_file()
    
        #print('-'*20)

        if linear:
            self.qrot = self.QROT_linear()
        else:
            self.qrot = self.QROT()
        
        #print('Q rotation: {0}'.format(self.qrot))

        self.qvib = self.QVIB(lower_bound = qlb)
        #print('Q vibration: {0}'.format(self.qvib))

        self.qtr = self.QTR()
        #print('Q translational: {0}'.format(self.qtr))

        self.Q = self.qtr * self.qvib * self.qrot
        
        #print('-'*20)

    def QROT_linear(self):
        for index, line in enumerate(self.data):
            if "THE MOMENTS OF INERTIA" in line:
                moments_amu = [float(_) for _ in self.data[index+1].split()]
                moments = [float(_) * amu * alu**2 for _ in self.data[index+1].split()]
    
        moment = [x for x in moments if x != 0][0]

        return 8 * np.pi**2 * moment * k * self.temperature / h**2

    def QROT(self):
        for index, line in enumerate(self.data):
            if "THE MOMENTS OF INERTIA" in line:
                moments_amu = [float(_) for _ in self.data[index+1].split()]
                moments = [float(_) * amu * alu**2 for _ in self.data[index+1].split()]
    
        return 8 * np.pi**2 * (8 * np.pi**3 * reduce(lambda x, y: x * y, moments))**(0.5) * (k * self.temperature)**(1.5) / h**3

    def read_file(self):
        with open(self.filename, mode = 'r') as inputfile:
            return inputfile.readlines()
    
    def QVIB(self, lower_bound = 7):
        frequencies = []
        
        for index, line in enumerate(self.data):
            if "FREQUENCY" in line:
                temp = [_ for _ in line.split()[1:]]

                for t in temp:
                    if self.is_float(t):
                        if float(t) > lower_bound:
                            frequencies.append(float(t))

        #print('frequencies: {0}'.format(frequencies))

        qvib = 1
        for freq in frequencies:
            qvib *= 1 / (1 - np.exp(- h * c * freq / (k * self.temperature)))
        
        return qvib

    def QTR(self):
        _ = self.m * amu
        return (2 * np.pi * _ * k * self.temperature / h**2)**(1.5)

    @staticmethod
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

temperatures = np.linspace(1200, 1500, 100)
constants_my = []
constants_article = []

for temperature in temperatures:
    print('TEMPERATURE: {0}'.format(temperature))

    c3h6_parser = QParser(filename = '../C3H6/c3h6.out', m = 42.0, temperature = temperature, qlb = 20)
    q_c3h6 = c3h6_parser.Q
     #print('Q C3H6: {0}'.format(q_c3h6))

    hf_parser = QParser(filename = '../HF/1_opt_hf.out', m = 20.0, temperature = temperature, linear = True, qlb = 20)
    q_hf = hf_parser.Q
     #print('Q HF: {0}'.format(q_hf))

    transition_parser =QParser(filename = '../saddle_point/sad.out', m = 62.0, temperature = temperature)
    q_transition = transition_parser.Q
    #print('Q transition_state: {0}'.format(q_transition))

    c3h7f_parser = QParser(filename = '../C3H7F/opt.out', m = 62.0, temperature = temperature, qlb = 40)
    q_c3h7f = c3h7f_parser.Q
    #print('Q c3h7f: {0}'.format(q_c3h7f))

    ZPE_c3h6 = 209.529677 * 10**3 # J / MOL at 0K
    ZPE_hf = 24.424847 * 10**3 # J / MOL at 0K
    ZPE_transition = 233.248230 * 10**3 # J / MOL at 0K
    ZPE_c3h7f = 252.554058 * 10**3 # J / MOL

    E_c3h6 = -117.916 # hartree
    E_hf = -100.427 # hartree
    E_transition = -218.278 # hartree
    E_c3h7f = -218.367 # hartree

    delta_Ef = (E_transition - E_c3h6 - E_hf) * hatoj + ZPE_transition - ZPE_c3h6 - ZPE_hf
    print('FORWARD Delta E: {0} J/mol'.format(delta_Ef))
    print('FORWARD Delta E: {0} kcal/mol'.format(delta_Ef / 4186.0))
    
    delta_Eb = (E_transition - E_c3h7f) * hatoj + ZPE_transition - ZPE_c3h7f
    delta_Eb_article = 56.57 * 4186.0 # J /mol

    #print('BACKWARD Delta E: {0} J/mol'.format(delta_Eb))
    #print('BACKWARD Delta E: {0} kcal/mol'.format(delta_Eb / 4186.0))

    #forward_reaction_rate = avogadro * k * temperature / h * q_transition / (q_c3h6 * q_hf) * np.exp(- delta_Ef / (R * temperature)) * 10**3 # comes from volume m3 -> l
    #print('FORWARD reaction rate: {0}'.format(forward_reaction_rate))

    backward_reaction_rate_my = k * temperature / h * q_transition / q_c3h7f * np.exp(- delta_Eb / (R * temperature)) 
    backward_reaction_rate_article = k * temperature / h * q_transition / q_c3h7f * np.exp(- delta_Eb_article / (R * temperature)) 
    #print('BACKWARD reaction rate: {0}'.format(backward_reaction_rate))
    
    constants_my.append(backward_reaction_rate_my)
    constants_article.append(backward_reaction_rate_article)

plt.plot(temperatures, constants_my, '--', color = 'blue')
plt.plot(temperatures, constants_article, '--', color = 'red')
plt.grid()

exp_temps = [1250 + 25 * i for i in range(10)]
exp_consts = [2.29 * 10**13 * np.exp(-225000.0 / (8.31 * temp)) for temp in exp_temps]

for t, c in zip(exp_temps, exp_consts):
    print('temperature: {0}; const: {1}'.format(t, c))
    plt.scatter(t, c, marker = '*', color = 'k')

blue_patch = mpatches.Patch(color = 'b', label = 'Calculated activation energy = 51.21kcal/mol')
red_patch = mpatches.Patch(color = 'r', label = 'Article activation energy = 56.57kcal/mol')
black_patch = mpatches.Patch(color = 'k', label = 'NIST')

plt.legend(handles = [blue_patch, red_patch, black_patch])

plt.show()

