'''
Simple driving script to run the Robot Dance model.
'''

print('Loading modules...')
import os
import os.path as path
from optparse import OptionParser
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import scipy.stats as stats
from timeit import default_timer as timer
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 7
print('Loading modules... Ok!')

import prepare_data

class SimpleTimeSeries:
    """Simple time series of one or two steps.
    """
    def __init__(self, rhomin, rhomax, c0, c1, phi1, phi2, sigmaw, initialm1, initial0):
        self.rhomin, self.rhomax = rhomin, rhomax
        self.C0, self.C1 = np.array([c0, 0]), np.array([c1, 0])
        self.A = np.array([[phi1, phi2], [1, 0]])
        self.sigmaw = sigmaw
        self.initial = np.array([initial0, initialm1])
        self.delta = rhomax - rhomin
        self.reset()
    

    def iterate(self, random=False):
        self.t += 1
        self.theta.append(self.Ak[0, 0])
        self.Ak = self.A @ self.Ak
        self.state = self.C0 + self.C1*self.t + self.A @ self.state
        if random:
            self.state[0] += np.random.normal(scale=self.sigmaw)
        return self.rhomin + self.delta*self.state[0]


    def reset(self):
        self.t = 0
        self.theta = []
        self.Ak = np.array([[1.0, 0], [0, 1]])
        self.state = (self.initial - self.rhomin) / self.delta
        


# To use PyJulia
print('Loading Julia library...')
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main as Julia
print('Loading Julia library... Ok!')
print('Loading Robot-dance Julia module...')
Julia.eval('include("robot_dance.jl")')
print('Loading Robot-dance Julia module... Ok!')


def get_options():
    '''Get options with file locations from command line.
    '''
    parser = OptionParser()
    parser.add_option("--basic_parameters", dest="basic_prm",
                      default=path.join("data", "basic_parameters.csv"),
                      help="Basic parameters of the SEIR model [default: %default]")
    parser.add_option("--cities_data", dest="cities_data",
                      default=path.join("data", "cities_data.csv"),
                      help="Population and initial state of the cities [default: %default]")
    parser.add_option("--pre_cities_data", dest="pre_cities_data",
                      default=path.join("data", "pre_cities_data.csv"),
                      help="Population and initial state of the cities [default: %default]")
    parser.add_option("--mobility_matrix", dest="mobility_matrix",
                      default=path.join("data", "mobility_matrix.csv"),
                      help="Mobility information [default: %default]")
    parser.add_option("--target", dest="target",
                      default=path.join("data", "target.csv"),
                      help="Maximal infected allowed [default: %default]")
    parser.add_option("--hammer_data", dest="hammer_data",
                      default=path.join("data", "hammer_data.csv"),
                      help="Hammer duration and level [default: %default]")
    options, dummy_args = parser.parse_args()
    return options


def read_data(options, verbosity=0):
    '''Read data from default files and locations.
    '''
    if path.exists(options.basic_prm):
        basic_prm = pd.read_csv(options.basic_prm, header=None, index_col=0, squeeze=True)
    else:
        if verbosity > 0:
            print("The file basic_parameters.csv is missing.")
            print("Using one with the default values from the report.")
        basic_prm = prepare_data.save_basic_parameters()
        
    if path.exists(options.cities_data):
        cities_data = pd.read_csv(options.cities_data, index_col=0)
    else:
        cities_data = prepare_data.compute_initial_condition_evolve_and_save(
            basic_prm, None, [], 0, 1, options.pre_cities_data)

    if path.exists(options.target):
        target = pd.read_csv(options.target, index_col=0)
    else:
        if verbosity > 0:
            print("Target for infected does not exits, usint 1%.")
        ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
        target = 0.01*np.ones((ncities, ndays))
        target = prepare_data.save_target(cities_data, target)

    
    if path.exists(options.mobility_matrix):
        mob_matrix = pd.read_csv(options.mobility_matrix, index_col=0)
        assert np.alltrue(mob_matrix.index == cities_data.index), \
            "Different cities in cities data and mobility matrix."
    else:
        ncities = len(cities_data)
        mob_matrix = pd.DataFrame(data=np.zeros((ncities, ncities)), 
            index=cities_data.index, columns=cities_data.index)
        mob_matrix["out"] = np.zeros(ncities)

    if path.exists(options.hammer_data):
        if verbosity > 0:
            print('Reading hammer data...')
        hammer_data = pd.read_csv(options.hammer_data, index_col=0)
        ncities = len(cities_data)
        assert len(hammer_data.index) == ncities, \
            "Different cities in cities data and hammer data"
        if verbosity > 0:
            print('Reading hammer data... Ok!')
    else:
        if verbosity > 0:
            print('Hammer data not found. Using default values')
        hammer_data = prepare_data.save_hammer_data(cities_data)

    return basic_prm, cities_data, mob_matrix, target, hammer_data


def prepare_optimization(basic_prm, cities_data, mob_matrix, target, hammer_data, 
    force_dif=1, pools=None, verbosity=0):
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
    if force_dif is 1:
        force_dif = np.ones((ncities, ndays))

    # Chage ratios in matrix Mt to be in respect to the origin
    population = cities_data["population"].values
    Mt = mob_matrix.values[:,:-1]
    Mt = (Mt.T).copy()
    for c in range(ncities):
        for k in range(ncities):
            Mt[k, c] *= population[k]/population[c]

    Julia.tinc = basic_prm["tinc"]
    Julia.tinf = basic_prm["tinf"]
    Julia.time_icu = basic_prm["time_icu"]
    Julia.need_icu = basic_prm["need_icu"]
    Julia.alternate = basic_prm["alternate"]
    Julia.rep = basic_prm["rep"]
    Julia.s1 = cities_data["S1"].values
    Julia.e1 = cities_data["E1"].values
    Julia.i1 = cities_data["I1"].values
    Julia.r1 = cities_data["R1"].values
    Julia.availICU = cities_data["icu_capacity"]
    Julia.population = population
    Julia.out = mob_matrix["out"].values
    Julia.M = mob_matrix.values[:, :-1]
    Julia.Mt = Mt
    Julia.ndays = ndays
    Julia.target = target.values
    Julia.min_level = basic_prm["min_level"]
    Julia.force_dif = force_dif
    Julia.hammer_duration = hammer_data["duration"].values
    Julia.hammer_level = hammer_data["level"].values
    Julia.verbosity = verbosity
    Julia.window = basic_prm["window"]
    if pools is None:
        Julia.eval("pools = [[c] for c in 1:length(s1)]")
    else:
        Julia.pools = pools
    Julia.eval("""
        prm = SEIR_Parameters(tinc, tinf, rep, ndays, time_icu, need_icu, alternate, 
                                s1, e1, i1, r1, availICU, window, out, sparse(M), sparse(Mt))
        m = window_control_multcities(prm, population, target, force_dif, hammer_duration, 
                                      hammer_level, min_level, pools, verbosity);
    """)

    # Check if there is a ramp parameter (delta_rt_max)
    # If so, add ramp constraints to the model
    if 'delta_rt_max' in basic_prm:
        Julia.delta_rt_max = basic_prm["delta_rt_max"]
        Julia.verbosity = verbosity
        Julia.eval("""
            m = add_ramp(m, prm, hammer_duration, delta_rt_max, verbosity)
        """)

def compute_need_icu(ts_parameters, basic_prm):
    p = 0.1
    F1p = stats.norm.ppf(1.0 - p)
    time_series = SimpleTimeSeries(*ts_parameters)
    need_icu = [time_series.iterate() for i in range(int(basic_prm["ndays"]))]
    # TODO: not sure p should be hard coded
    theta = np.array(time_series.theta).copy()
    for i in range(int(basic_prm["ndays"])):
        need_icu[i] += F1p*time_series.sigmaw*time_series.delta*np.sqrt((theta[:i + 1]**2).sum())
    return need_icu


def find_feasible_hammer(basic_prm, cities_data, mob_matrix, target, hammer_data, 
    out_file=None, incr_all=False, verbosity=0):
    """Find hammer durations for each city such that the optimization problem will 
    (hopefully) be feasible
    """
    # TODO: This has to be transformed into parameters
    rmsp = pool = np.array([9, 15, 16, 17, 18, 19]) - 1
    if basic_prm["time_icu"] == 7:
        need_icu_sp = compute_need_icu([0.01099859, 0.02236023, 0.00370254, 0.0, 1.79119571, -0.80552926, 
            np.sqrt(0.00034005), 0.011644768910252, 0.011221496171591], basic_prm)
        need_icu_notsp = compute_need_icu([0.0076481, 0.0218084, 0.00367839, 0.0, 1.81361379, -0.82550856, 
            np.sqrt(8.028E-05), 0.007907216664912, 0.007721801045322], basic_prm)
    elif basic_prm["time_icu"] == 11:
        need_icu_sp = compute_need_icu([0.0074335, 0.01523406, -0.00186355, 0.0, 1.67356018, -0.68192908, 
            np.sqrt(0.00023883), 0.007682840158843, 0.007536060983504], basic_prm)
        need_icu_notsp = compute_need_icu([0.00520255, 0.01532709, 0.00044498, 0.0, 1.75553282, -0.76360711,
            np.sqrt(3.567E-05), 0.005426447471187, 0.005282217308748], basic_prm)
    else:
        raise NotImplementedError


    if verbosity >= 1:
        print('Checking if initial hammer is long enough...')
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])

    M = mob_matrix.values[:,:-1]
    out = mob_matrix["out"].values

    tspan = (0,ndays)
    teval = np.arange(0, ndays + 0.01, 1) # 1 day discretization
    y0 = cities_data["S1"].values
    y0 = np.append(y0, cities_data["E1"].values)
    y0 = np.append(y0, cities_data["I1"].values)
    y0 = np.append(y0, cities_data["R1"].values)

    min_rt = basic_prm["min_level"]

    # Hammer data
    hammer_duration = hammer_data["duration"].values
    hammer_level = hammer_data["level"].values

    iter = 0
    feas_model = False
    t_total1 = timer()
    while feas_model == False:
        t_solve1 = timer()
        sol = solve_ivp(_robot_dance_hammer, tspan, y0, t_eval=teval, args=(
                basic_prm["tinc"], basic_prm["tinf"], ncities, M, out, min_rt, 
                hammer_level, hammer_duration
            )
        )
        t_solve2 = timer()
        if verbosity >= 2:
            print(f'Time to simulate: {t_solve2-t_solve1:.2g}')

        tsim = sol.t
        isim = sol.y[2*ncities:3*ncities]

        # Get the max number of infected after hammer (usually it's the number
        # of infected immediately after hammer, but this might not work if
        # hammer_duration = 0) (so let's use a safer implementation)
        i_after_hammer = np.zeros(ncities)
        target_hammer = np.zeros(ncities)
        for c in range(ncities):
            if c in rmsp:
                need_icu = need_icu_sp
            else:
                need_icu = need_icu_notsp

            target_hammer[c] = 0.7*target.iloc[c][hammer_duration[c] + 1]*cities_data.iloc[c]["icu_capacity"]
            i_after_hammer[c] = basic_prm["time_icu"]*need_icu[hammer_duration[c]]*max(
                isim[c][hammer_duration[c] + 1:])/basic_prm["tinf"]

        feas_model = True
        for c in range(ncities):
            if i_after_hammer[c] > target_hammer[c]:
                if verbosity >= 2:
                    print(f'{cities_data.index[c]} violates number of infected after {hammer_duration[c]} days of hammer (level {hammer_level[c]}): Infected {i_after_hammer[c]:.2g} (target {target_hammer[c]})')
                feas_model = False
            else:
                if verbosity >= 2:
                    print(f'{cities_data.index[c]} is fine after {hammer_duration[c]} days of hammer (infected = {i_after_hammer[c]:.2e}, target = {target_hammer[c]:.2e})')

        if feas_model == False:
            # There is at least one city violating the target after hammer
            if incr_all == False: # Increase hammer_duration only for the city that violates the target the most
                c_i_max = np.argmax(i_after_hammer-target_hammer)
                if verbosity >= 2:
                    print(f'City most distant from target after hammer: {cities_data.index[c_i_max]}')
                if hammer_duration[c_i_max] == ndays:
                    raise ValueError(f'Impossible to get a feasible model (hammer_duration for {cities_data.index[c]} is equal to the simulation horizon). Try increasing ndays or decreasing hammer_level')
                hammer_duration[c_i_max] += basic_prm["window"]
                if verbosity >= 2:
                    print(f'Increasing hammer duration of {cities_data.index[c_i_max]} to {hammer_duration[c_i_max]} days')
            else: # Increase hammer_duration of all cities that violate the target
                for c in range(ncities):
                    if i_after_hammer[c] > target_hammer[c]:
                        if hammer_duration[c] == ndays:
                            raise ValueError(f'Impossible to get a feasible model (hammer_duration for {cities_data.index[c]} is equal to the simulation horizon). Try increasing ndays or decreasing hammer_level')
                        hammer_duration[c] += basic_prm["window"]
                        if verbosity >= 2:
                            print(f'Increasing hammer duration of {cities_data.index[c]} to {hammer_duration[c]} days')
            if verbosity >= 2:    
                print('')
        iter += 1

    t_total2 = timer()
    if verbosity >= 1:
        print('')
        print(f'Number of iterations: {iter}')
        print(f'Total time: {t_total2-t_total1} s')
        print()
        print("Hammer data")
        print(hammer_data)

    if out_file is not None:
        if verbosity >= 1:
            print('Saving hammer data file')
        hammer_data.to_csv(out_file)

    if verbosity >= 1:
        print('Checking if initial hammer is long enough... Ok!')


def check_error_optim(basic_prm, cities_data, mob_matrix, dir_output, verbosity=0):
    """ Checks error between optimization and simulation
    """
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])

    M = mob_matrix.values[:,:-1]
    out = mob_matrix["out"].values

    tspan = (1,ndays)
    teval = range(1,ndays+1)
    y0 = cities_data["S1"].values
    y0 = np.append(y0, cities_data["E1"].values)
    y0 = np.append(y0, cities_data["I1"].values)
    y0 = np.append(y0, cities_data["R1"].values)

    Julia.eval("s = value.(m[:s]); e = value.(m[:e]); i = value.(m[:i]); r = value.(m[:r])")
    Julia.eval("rt = expand(value.(m[:rt]), prm)")
    t_in = teval
    rt_in = Julia.rt
    if verbosity >= 1:
        print('Simulating robot-dance control...')
    sol = solve_ivp(_robot_dance_simul, tspan, y0, t_eval=teval, args=(basic_prm["tinc"], \
                                                                        basic_prm["tinf"], \
                                                                        ncities, \
                                                                        M, \
                                                                        out, \
                                                                        t_in, \
                                                                        rt_in))
    if verbosity >= 1:
        print('Simulating robot-dance control... Ok!')
    s_sim = sol.y[:ncities]
    e_sim = sol.y[ncities:2*ncities]
    i_sim = sol.y[2*ncities:3*ncities]
    r_sim = sol.y[3*ncities:]
    
    if verbosity >= 1:
        print('Plotting errors...')
    for (i,c) in enumerate(cities_data.index):
        fig = plt.figure()
        plt.plot(Julia.s[i], label="robot-dance")
        plt.plot(s_sim[i], label="simulation")
        plt.legend()
        plt.title(f'{c}, Susceptible')
        plt.savefig(f'{dir_output}/{c}_s.png')

        fig = plt.figure()
        plt.plot(Julia.e[i], label="robot-dance")
        plt.plot(e_sim[i], label="simulation")
        plt.legend()
        plt.title(f'{c}, Exposed')
        plt.savefig(f'{dir_output}/{c}_e.png')

        fig = plt.figure()
        plt.plot(Julia.i[i], label="robot-dance")
        plt.plot(i_sim[i], label="simulation")
        plt.legend()
        plt.title(f'{c}, Infected')
        plt.savefig(f'{dir_output}/{c}_i.png')

        fig = plt.figure()
        plt.plot(Julia.r[i], label="robot-dance")
        plt.plot(r_sim[i], label="simulation")
        plt.legend()
        plt.title(f'{c}, Removed')
        plt.savefig(f'{dir_output}/{c}_r.png')
    if verbosity >= 1:
        print('Plotting errors... Ok!')

    fig = plt.figure()
    for (i,c) in enumerate(cities_data.index):
        plt.plot(rt_in[i], label=c)
    plt.legend()
    plt.grid()
    plt.title('Control rt')
    plt.savefig(f'{dir_output}/rt.png')

    rt_diff = []
    for (i,c) in enumerate(cities_data.index):
        rt_diff.append(np.diff(rt_in[i]))
    
    fig = plt.figure()
    for (i,c) in enumerate(cities_data.index):
        plt.plot(rt_diff[i], label=c)
    plt.legend()
    plt.grid()
    plt.title('Diff rt')
    plt.savefig(f'{dir_output}/diff_rt.png')

    plt.show()

    if verbosity >= 1:
        print('Saving errors table...')
    df = pd.DataFrame(columns=['s_norm_1', 'e_norm_1', 'i_norm_1', 'r_norm_1', 's_norm_inf', 'e_norm_inf', 'i_norm_inf', 'r_norm_inf'], index=cities_data.index)
    for (i,c) in enumerate(cities_data.index):
        df.loc[c, 's_norm_1'] = np.linalg.norm(s_sim[i]-Julia.s[i], ord=1)
        df.loc[c, 'e_norm_1'] = np.linalg.norm(e_sim[i]-Julia.e[i], ord=1)
        df.loc[c, 'i_norm_1'] = np.linalg.norm(i_sim[i]-Julia.i[i], ord=1)
        df.loc[c, 'r_norm_1'] = np.linalg.norm(r_sim[i]-Julia.r[i], ord=1)
        df.loc[c, 's_norm_inf'] = np.linalg.norm(s_sim[i]-Julia.s[i], ord=np.inf)
        df.loc[c, 'e_norm_inf'] = np.linalg.norm(e_sim[i]-Julia.e[i], ord=np.inf)
        df.loc[c, 'i_norm_inf'] = np.linalg.norm(i_sim[i]-Julia.i[i], ord=np.inf)
        df.loc[c, 'r_norm_inf'] = np.linalg.norm(r_sim[i]-Julia.r[i], ord=np.inf)
    df.to_csv(f'{dir_output}/error_discretization.csv')
    if verbosity >= 1:
        print('Saving errors table... Ok!')
        

def _robot_dance_only_eqs(s,e,i,r,rt,tinc,tinf,ncities,M,out):
    """SEIR equations for the robot-dance model
    """
    alpha = 2/3

    enter = np.zeros(ncities)
    for c1 in range(ncities):
        for c2 in range(ncities):
            enter[c1] += M[c2,c1]*(1-i[c2])

    p_day = np.zeros(ncities)
    for c in range(ncities):
        p_day[c] = (1-out[c]) + out[c]*i[c] + enter[c]

    t1 = np.zeros(ncities)
    for c1 in range(ncities):
        for c2 in range(ncities):
            t1[c1] += rt[c2]*M[c1,c2]*s[c1]*i[c2]/p_day[c2]

    ds_day = -1/tinf * alpha * (rt * (1-out) * s * i / p_day + t1)
    ds_night = -1/tinf * (1-alpha) * (rt * s * i)
    ds = ds_day + ds_night

    de = -ds - 1/tinc*e
    di = 1/tinc*e - 1/tinf*i
    dr = 1/tinf*i

    dy = np.array([ds,de,di,dr]).flatten()
    return dy


def _robot_dance_simul(t,y,tinc,tinf,ncities,M,out, t_in, rt_in):
    """SEIR equations for the robot-dance model with control rt given by the optimization model
    """
    s = y[:ncities]
    e = y[ncities:2*ncities]
    i = y[2*ncities:3*ncities]
    r = y[3*ncities:]

    # Interpolate rt for each city for the current t
    rt = np.zeros(ncities)
    for c in range(ncities):
        rt[c] = np.interp(t, t_in, rt_in[c])

    dy = _robot_dance_only_eqs(s,e,i,r,rt,tinc,tinf,ncities,M,out)
    return dy


def _robot_dance_hammer(t,y,tinc,tinf,ncities,M,out,min_rt,hammer_level,hammer_duration):
    """SEIR equations for the robot-dance model with initial hammer and min_rt later
    # (used to get a hammer duration such that the optimization problem will be feasible)
    """
    s = y[:ncities]
    e = y[ncities:2*ncities]
    i = y[2*ncities:3*ncities]
    r = y[3*ncities:]

    rt = np.zeros(ncities)
    for c in range(ncities):
        if t <= hammer_duration[c]:
            # Enforce hammer in the initial period
            rt[c] = hammer_level[c]
        else:
            # Enforce min rt (not as strict as hammer) for the rest of horizon
            rt[c] = min_rt

    dy = _robot_dance_only_eqs(s,e,i,r,rt,tinc,tinf,ncities,M,out)
    return dy


def save_result(cities_names, filename):
    """Save the result of a run for further processing.
    """
    Julia.eval("s = value.(m[:s]); e = value.(m[:e]); i = value.(m[:i]); r = value.(m[:r])")
    Julia.eval("rt = expand(value.(m[:rt]), prm)")
    df = []
    for i in range(len(cities_names)):
        c = cities_names[i]
        df.append([c, "s"] + list(Julia.s[i, :])) 
        df.append([c, "e"] + list(Julia.e[i, :])) 
        df.append([c, "i"] + list(Julia.i[i, :])) 
        df.append([c, "r"] + list(Julia.r[i, :])) 
        df.append([c, "rt"] + list(Julia.rt[i, :])) 
    df = pd.DataFrame(df, columns=["City", "Variable"] + list(range(len(Julia.s[0,:]))))
    df.set_index(["City", "Variable"], inplace=True)
    df.to_csv(filename)
    return df


def optimize_and_show_results(basic_prm, figure_file, data_file, cities_data, verbosity=0):
    """Optimize and save figures and data for further processing.
    """

    large_cities = cities_data.index
    if verbosity >= 1:
        print('Solving Robot-dance...')

    Julia.eval("""
        optimize!(m)
        pre_rt = value.(m[:rt]); i = value.(m[:i])
        rt = expand(pre_rt, prm)
    """)

    if verbosity >= 1:
        print('Solving Robot-dance... Ok!')

    bins = [0]
    bins.extend(plt.linspace(1.0, 0.95*basic_prm["rep"], 5))
    bins.append(basic_prm["rep"])

    stats = pd.DataFrame(index=large_cities)
    changes_rt = []
    for (i, c) in enumerate(large_cities):
        changes_rt.append(len(np.diff(Julia.rt[i]).nonzero()[0]) + 1)
    stats["Rt changes"] = changes_rt

    i_avg, max_i = [], []
    for (i, c) in enumerate(large_cities):
        maximum = 100*Julia.i[i, Julia.hammer_duration[i]:].max()
        average = 100*sum(Julia.i[i])/len(Julia.i[i])
        max_i.append(f"{maximum:.3f}%")
        i_avg.append(f"{average:.3f}%")
    stats["Avg. I"] = i_avg
    stats["Max I"] = max_i
        
    total, mean = [], []
    for (i,c) in enumerate(large_cities):
        rt = Julia.rt[i]
        inds = np.nonzero(rt >= bins[-2])[0]
        count_open_total = len(inds)
        thresh_open = np.nonzero(np.diff(inds) > 1)[0] + 1
        thresh_open = np.insert(thresh_open, 0, 0)
        thresh_open = np.append(thresh_open, len(inds))
        count_open = np.diff(thresh_open)
        total.append(count_open_total)
        mean.append(np.mean(count_open))
    stats["Open"] = total
    stats["Mean open"] = mean

    total, mean = [], []
    for (i,c) in enumerate(large_cities):
        rt = Julia.rt[i]
        inds = np.nonzero(rt < bins[2])[0]
        count_open_total = len(inds)
        thresh_open = np.nonzero(np.diff(inds) > 1)[0] + 1
        thresh_open = np.insert(thresh_open, 0, 0)
        thresh_open = np.append(thresh_open, len(inds))
        count_open = np.diff(thresh_open)
        total.append(count_open_total)
        mean.append(np.mean(count_open))
    stats["Closed"] = total
    stats["Mean closed"] = mean

    if verbosity >= 1:
        print()
        print("Statistics")
        print(stats)

    # Before saving anything, check if directory exists
    # Lets assume all output files are in the same directory
    dir_output = path.split(figure_file)[0]
    if not path.exists(dir_output):
        os.makedirs(dir_output)

    if verbosity >= 1:
        print('Saving output files...')
    
    result = save_result(large_cities, data_file)
    
    if verbosity >= 1:
        print('Saving output files... Ok!')

    if verbosity >= 1:
        print("Ploting result...")

    plot_result(basic_prm, result, figure_file, Julia.hammer_duration, 
        cities_data["start_date"][0])
    plt.savefig(figure_file, dpi=150, bbox_inches='tight')

    if verbosity >= 1:
        print("Ploting result... OK!")

    return stats


def plot_result(basic_prm, result, figure_file, hammer_duration, start_date=None,
    subset=None):
    """Plot result in a single figure.
    """
    
    # Get data
    if subset is None:
        cities = result.index.get_level_values(0).unique()
    else:
        cities = subset
        
    max_city_len = np.max([len(c) for c in cities])
    window = int(basic_prm["window"])
    if start_date is not None:
        start_date = pd.Timestamp(start_date)

    # Find the maximal infected rates
    ncities = len(cities)
    max_i = np.zeros((ncities, 2))
    for j in range(ncities):
        city_name = cities[j]
        i, rt = result.loc[city_name, "i"], result.loc[city_name, "rt"]
        max_i[j, 0] = i.max()
        end_hammer = hammer_duration[j] 
        max_i[j, 1] = i.iloc[end_hammer:].max()
                
    # Create figure    
    fig = plt.figure(figsize=(15, 1*ncities), constrained_layout=False)

    gs = gridspec.GridSpec(ncities, 2, height_ratios=max_i[:, 0], width_ratios=[0.82, 0.18],
        hspace=0, wspace=0)
    # Colors for rt
    bins = [0]
    bins.extend(plt.linspace(1.0, 0.95*basic_prm["rep"], 5))
    bins.append(basic_prm["rep"])
    bins = np.array(bins)
    colors = ['orangered','darkorange','gold','blue','green','aliceblue']
    levels = ['Severe','High','Elevated','Moderate','Low','Open']
    
    ax = plt.subplot(gs[:, 1])
    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=levels[i]) for i in range(len(colors))]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_axis_off()

    for j in range(ncities):
        # Get data
        city_name = cities[j]
        i, rt = result.loc[city_name, "i"], result.loc[city_name, "rt"]
        ndays = len(i) - 1

        # Prepare figure
        ax = plt.subplot(gs[j, 0])

        # Plot infected 
        ax.plot([0, ndays], [max_i[j, 1], max_i[j, 1]], color="k", alpha=0.15)
        ax.plot(i, color="k")
        # # Show the basolute maximal level before hammer
        # if max_i[j, 0] >= 1.2*max_i[j, 1]:
        #     ax.plot([0, ndays], [max_i[j, 0], max_i[j, 0]], color="k", alpha=0.15)
        
        # Plot target R0(t)
        for d in range(0, len(rt) - 1, window):
            color_ind = np.searchsorted(bins, rt.iloc[d]) - 1
            r = Rectangle((d, 0), min(window, ndays - d), 1.1*max_i[j, 0], 
                          color=colors[color_ind])
            ax.add_patch(r)

        # Set up figure
        ax.set_xticks([])
        ax.set_xticklabels([])
        ylabel_format = "{{:>{}s}}".format(max_city_len)
        ax.set_ylabel(ylabel_format.format(city_name), rotation=0, horizontalalignment="left", labelpad=80)
        if max_i[j, 0] >= 1.2*max_i[j, 1]:
            # # Show absoltute maximal level before hammer
            # ax.set_yticks(max_i[j, :])
            # ax.set_yticklabels(["{:.2f}%".format(100*max_i[j, k]) for k in [0, 1]])
            ax.set_yticks([max_i[j, 1]])
            ax.set_yticklabels(["{:.2f}%".format(100*max_i[j, 1])])
        else:
            ax.set_yticks([max_i[j, 1]])
            ax.set_yticklabels(["{:.2f}%".format(100*max_i[j, 1])])
            
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_right()
        # ax.tick_params(axis = "y", which = "both", left = False, right = False)

        ax.spines['left'].set_color("darkgrey")
        ax.spines['right'].set_color("darkgrey")
        ax.spines['top'].set_color("darkgrey")
        ax.spines['bottom'].set_color("darkgrey")
        #ax.spines['bottom'].set_visible(False)

        ax.set_xlim(0, ndays)
        ax.set_ylim(0, 1.1*max_i[j, 0])

        if j == 0:
            ax.set_title("Infection level and target rt")

    if start_date is None:
        ax.set_xticks(np.arange(0, ndays, 30))
    else:
        ticks = pd.date_range(start_date, start_date + ndays*pd.to_timedelta("1D"), freq="1MS")
        ticks = list(ticks)
        if ticks[0] <= start_date + pd.to_timedelta("10D"):
            ticks[0] = start_date
        else:
            ticks = [start_date] + ticks
        ax.set_xticks([(i - start_date).days for i in ticks])
        labels = [i.strftime('%d/%m/%Y') for i in ticks]
        ax.set_xticklabels(labels, rotation=45, ha='right')

    

def main():
    """Allow call from the command line.
    """
    verbosity = 1 # 0: print nothing, 1: print min info (with solver progress), 
                  # 2: even more detailed
    dir_output = "results"
    options = get_options()
    basic_prm, cities_data, mob_matrix, target, hammer_data = read_data(options)
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
    force_dif = np.ones((ncities, ndays))
    find_feasible_hammer(basic_prm, cities_data, mob_matrix, target, hammer_data, 
        out_file=options.hammer_data, incr_all=True, verbosity=verbosity)
    prepare_optimization(basic_prm, cities_data, mob_matrix, target, hammer_data, force_dif, 
        verbosity=verbosity)
    optimize_and_show_results(basic_prm, f"{dir_output}/cmd_res.png", 
        f"{dir_output}/cmd_res.csv", cities_data.index, verbosity=verbosity)
    # check_error_optim(basic_prm, cities_data, mob_matrix, dir_output)

if __name__ == "__main__":
    main()
