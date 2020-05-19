# Author: Luis Gustavo Nonato  -- <gnonato@icmc.usp.br>
# License: See LICENSE.md file in the repository.
# This is an implementation of a SEIR epidemic model

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class seir():
    def __init__(self,ndays):
        '''
        ndays: number of days to simulate the epidemic outbreak
        '''
        self.dt = 1.0e-2
        self.ndays = ndays

        self.R0 = 2.5
        self.Tinc = 5.2
        self.Tinf = 2.9


    def run(self,initial_condition,Rnew=1.0,social_dist_period=(0,0)):
        '''
        Rnew: social distancing intensity in the range [0,1], where 0 means no distancing and 1 total distancing
        social_dist_period: tuple (d0,dk) where d0 and dk are the starting and ending day of the social distancing (dk <= ndays)

        initial_condition: tuple (S0,E0,I0,R0) containing:
            S0: healthy population that can be contaminated
            E0: contaminated population with incubated virus (not transmitting the disease)
            I0: contaminated population transmitting the desease
            R0: recovered population
        '''
        initialc = initial_condition

        rt = self.R0
        solution = np.zeros((4,self.ndays))
        def _seir(t,y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return(np.asarray([-(rt/self.Tinf)*S*I, (rt/self.Tinf)*S*I - (1.0/self.Tinc)*E, (1.0/self.Tinc)*E - (1.0/self.Tinf)*I, (1.0/self.Tinf)*I]))

        if (social_dist_period[0] == social_dist_period[1]):
            SEIR = solve_ivp(_seir, [0, self.ndays], initialc, t_eval=np.arange(0, self.ndays, 1))
            solution[:] = SEIR.y
        else:
            SEIR = solve_ivp(_seir, [0, social_dist_period[0]], initialc, t_eval=np.arange(0, social_dist_period[0], 1))
            solution[:,:social_dist_period[0]] = SEIR.y

            rt = Rnew*self.R0
            SEIR = solve_ivp(_seir, [social_dist_period[0], social_dist_period[1]], (SEIR.y[0][-1],SEIR.y[1][-1],SEIR.y[2][-1],SEIR.y[3][-1]), t_eval=np.arange(social_dist_period[0], social_dist_period[1], 1))
            solution[:,social_dist_period[0]:social_dist_period[1]] = SEIR.y

            if social_dist_period[1] < self.ndays:
               rt = self.R0
               SEIR = solve_ivp(_seir, [social_dist_period[1], self.ndays], (SEIR.y[0][-1],SEIR.y[1][-1],SEIR.y[2][-1],SEIR.y[3][-1]), t_eval=np.arange(social_dist_period[1], self.ndays, 1))
               solution[:,social_dist_period[1]:] = SEIR.y


        return(solution)


class initial_condition():
    def __init__(self,data):
        self.ndays = data.shape[0]
        self.data = data
        self.Rt = 2.5
        self.Tinc = 5.2
        self.Tinf = 2.9

    def run(self,init0):

        def _seir(t,y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return(np.asarray([-(self.Rt/self.Tinf)*S*I, (self.Rt/self.Tinf)*S*I - (1.0/self.Tinc)*E, (1.0/self.Tinc)*E - (1.0/self.Tinf)*I, (1.0/self.Tinf)*I]))

        def loss(EI):
            f = solve_ivp(_seir, [0, self.ndays], (init0[0],EI[0],EI[1],init0[3]), t_eval=np.arange(0, self.ndays, 1))
            return(np.sqrt(np.mean((f.y[2] - self.data)**2)))

        optimal = minimize(
            loss,
            [init0[1],init0[2]],
            method='L-BFGS-B',
            bounds=[(0.0, 1.0), (0.0, 1.0)])

        E0, I0 = optimal.x
        return(E0,I0)
