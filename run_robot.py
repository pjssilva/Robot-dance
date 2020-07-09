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
from timeit import default_timer as timer
import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 7
print('Loading modules... Ok!')


import prepare_data

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


def read_data(options):
    '''Read data from default files and locations.
    '''
    if path.exists(options.basic_prm):
        basic_prm = pd.read_csv(options.basic_prm, header=None, index_col=0, squeeze=True)
    else:
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
        print('Reading hammer data...')
        hammer_data = pd.read_csv(options.hammer_data, index_col=0)
        print('Reading hammer data... Ok!')
    else:
        print('Hammer data not found. Using default values')
        hammer_data = prepare_data.save_hammer_data(cities_data)

    return basic_prm, cities_data, mob_matrix, target, hammer_data


def prepare_optimization(basic_prm, cities_data, mob_matrix, target, hammer_data, force_dif=1):
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
    if force_dif is 1:
        force_dif = np.ones((ncities, ndays))

    Julia.tinc = basic_prm["tinc"]
    Julia.tinf = basic_prm["tinf"]
    Julia.rep = basic_prm["rep"]
    Julia.s1 = cities_data["S1"].values
    Julia.e1 = cities_data["E1"].values
    Julia.i1 = cities_data["I1"].values
    Julia.r1 = cities_data["R1"].values
    Julia.population = cities_data["population"].values
    Julia.out = mob_matrix["out"].values
    Julia.M = mob_matrix.values[:, :-1]
    Julia.ndays = ndays
    Julia.target = target.values
    Julia.min_level = basic_prm["min_level"]
    Julia.force_dif = force_dif
    Julia.hammer_duration = hammer_data["duration"].values
    Julia.hammer_level = hammer_data["level"].values
    if basic_prm["window"] == 1:
        Julia.eval("""
            prm = SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, 1, out, sparse(M), 
                                  sparse(M'))
            m = control_multcities(prm, population, target, force_dif, hammer_duration, 
                                   hammer_level, min_level)
        """)
    else:
        Julia.window = basic_prm["window"]
        Julia.eval("""
            prm = SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, window, out, 
                                  sparse(M), sparse(M'))
            m = window_control_multcities(prm, population, target, force_dif, 
                                          hammer_duration, hammer_level, min_level);
        """);        


def find_feasible_hammer(basic_prm, cities_data, mob_matrix, target, hammer_data, options, incr_all=False, save_file=False):
    """Find hammer durations for each city such that the optimization problem will (hopefully) be feasible
    """
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])

    M = mob_matrix.values[:,:-1]
    out = mob_matrix["out"].values

    tspan = (0,ndays)
    teval = np.arange(0, ndays+0.01, 1) # 1 day discretization
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
        sol = solve_ivp(_robot_dance_eqs, tspan, y0, t_eval=teval, args=(basic_prm["tinc"], \
                                                                        basic_prm["tinf"], \
                                                                        ncities, \
                                                                        M, \
                                                                        out, \
                                                                        min_rt, \
                                                                        hammer_level, \
                                                                        hammer_duration))
        t_solve2 = timer()
        print(f'Time to simulate: {t_solve2-t_solve1}')

        tsim = sol.t
        isim = sol.y[2*ncities:3*ncities]

        # Get the max number of infected after hammer
        # (usually the number of infected immediately after hammer, but this might not work if hammer_duration = 0)
        # (so let's use the safest implementation)
        i_after_hammer = np.zeros(ncities)
        target_hammer = np.zeros(ncities)
        for c in range(ncities):
            target_hammer[c] = target.iloc[c][hammer_duration[c]]
            i_after_hammer[c] = max(isim[c][hammer_duration[c]:])

        feas_model = True
        for c in range(ncities):
            if i_after_hammer[c] > target_hammer[c]:
                print(f'{cities_data.index[c]} violates number of infected after {hammer_duration[c]} days of hammer (level {hammer_level[c]}): Infected {i_after_hammer[c]:.2g} (target {target_hammer[c]})')
                feas_model = False
            else:
                print(f'{cities_data.index[c]} is fine after {hammer_duration[c]} days of hammer (infected = {i_after_hammer[c]:.2g}, target = {target_hammer[c]})')

        if feas_model == False:
            # There is at least one city violating the target after hammer
            if incr_all == False: # Increase hammer_duration only for the city that violates the target the most
                c_i_max = np.argmax(i_after_hammer-target_hammer)
                print(f'City most distant from target after hammer: {cities_data.index[c_i_max]}')
                if hammer_duration[c_i_max] == ndays:
                    raise ValueError(f'Impossible to get a feasible model (hammer_duration for {cities_data.index[c]} is equal to the simulation horizon). Try increasing ndays or decreasing hammer_level')
                hammer_duration[c_i_max] += basic_prm["window"]
                print(f'Increasing hammer duration of {cities_data.index[c_i_max]} to {hammer_duration[c_i_max]} days')
            else: # Increase hammer_duration of all cities that violate the target
                for c in range(ncities):
                    if i_after_hammer[c] > target_hammer[c]:
                        if hammer_duration[c] == ndays:
                            raise ValueError(f'Impossible to get a feasible model (hammer_duration for {cities_data.index[c]} is equal to the simulation horizon). Try increasing ndays or decreasing hammer_level')
                        hammer_duration[c] += basic_prm["window"]
                        print(f'Increasing hammer duration of {cities_data.index[c]} to {hammer_duration[c]} days')
            print('')
        iter += 1

    t_total2 = timer()
    print('')
    print(f'Number of iterations: {iter}')
    print(f'Total time: {t_total2-t_total1} s')
    print('Hammer duration')
    for c in range(ncities):
        print(f'{cities_data.index[c]}: {hammer_duration[c]} days')

    if save_file == True:
        hammer_data.to_csv(options.hammer_data)


def _robot_dance_eqs(t,y,tinc,tinf,ncities,M,out,min_rt,hammer_level,hammer_duration):
    """SEIR equations for the robot-dance model
    """
    s = y[:ncities]
    e = y[ncities:2*ncities]
    i = y[2*ncities:3*ncities]
    r = y[3*ncities:]
    alpha = 2/3

    rt = np.zeros(ncities)
    for c in range(ncities):
        if t <= hammer_duration[c]:
            # Enforce hammer in the initial period
            rt[c] = hammer_level[c]
        else:
            # Enforce min rt (not as strict as hammer) for the rest of horizon
            rt[c] = min_rt

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


def optimize_and_show_results(i_fig, rt_fig, data_file, large_cities):
    """Optimize and save figures and data for further processing.
    """

    Julia.eval("""
        optimize!(m)
        pre_rt = value.(m[:rt]); i = value.(m[:i])
        rt = expand(pre_rt, prm)
    """)

    print('')
    print('Number of rt changes in each city')
    for (i, c) in enumerate(large_cities):
        changes_rt = len(np.diff(Julia.rt[i]).nonzero()[0]) + 1
        print(f'{c}: {changes_rt}')

    print('')
    print('Average fraction of infected')
    for (i, c) in enumerate(large_cities):
        i_avg = sum(Julia.i[i])/len(Julia.i[i])
        print(f'{c}: {i_avg}')
        

    # Before saving anything, check if directory exists
    # Lets assume all output files are in the same directory
    dir_output = path.split(i_fig)[0]
    if not path.exists(dir_output):
        os.makedirs(dir_output)

    for i in range(len(large_cities)):
        plt.plot(Julia.rt[i, :], label=large_cities[i], lw=5, alpha=0.5)
    plt.legend()
    plt.title("Target reproduction rate")
    plt.savefig(rt_fig)

    plt.clf()
    for i in range(len(large_cities)):
        plt.plot(Julia.i[i, :], label=large_cities[i])
    plt.legend()
    plt.title("Infection level")
    plt.savefig(i_fig)

    save_result(large_cities, data_file)


def main():
    """Allow call from the command line.
    """
    options = get_options()
    basic_prm, cities_data, mob_matrix, target, hammer_data = read_data(options)
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
    force_dif = np.ones((ncities, ndays))
    find_feasible_hammer(basic_prm, cities_data, mob_matrix, target, hammer_data, options, incr_all=True, save_file=False)
    prepare_optimization(basic_prm, cities_data, mob_matrix, target, hammer_data, force_dif)
    optimize_and_show_results("results/cmd_i_res.png", "results/cmd_rt_res.png",
                              "results/cmd_res.csv", cities_data.index)

if __name__ == "__main__":
    main()
