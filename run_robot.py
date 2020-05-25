'''
Simple driving script to run the Robot Dance model.
'''

import os
import os.path as path
from optparse import OptionParser
import pandas as pd
import numpy as np
import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 7

import prepare_data

# To use PyJulia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main as Julia
Julia.eval('ENV["OMP_NUM_THREADS"] = 8')
Julia.eval('include("robot_dance.jl")')


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
    options, dummy_args = parser.parse_args()
    return options


def read_data(options):
    '''Read data from default files and locations.
    '''
    basic_prm = pd.read_csv(options.basic_prm, header=None, index_col=0, squeeze=True)
    if path.exists(options.cities_data):
        cities_data = pd.read_csv(options.cities_data, index_col=0)
    else:
        cities_data = prepare_data.compute_initial_condition_evolve_and_save(
            basic_prm, None, [], 0, 1, options.pre_cities_data)
    target = pd.read_csv(options.target, index_col=0)
    if path.exists(options.mobility_matrix):
        mob_matrix = pd.read_csv(options.mobility_matrix, index_col=0)
        assert np.alltrue(mob_matrix.index == cities_data.index), \
            "Different cities in cities data and mobility matrix."
    else:
        ncities = len(cities_data)
        mob_matrix = pd.DataFrame(data=np.zeros((ncities, ncities)), 
            index=cities_data.index, columns=cities_data.index)

    return basic_prm, cities_data, mob_matrix, target


def prepare_optimization(basic_prm, cities_data, mob_matrix, target, force_dif=1):
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
    Julia.hammer_duration = int(basic_prm["hammer_duration"])
    Julia.hammer_level = basic_prm["hammer_level"]
    Julia.min_level = basic_prm["min_level"]
    Julia.force_dif = force_dif
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
    return df.to_csv(filename)


def optimize_and_show_results(i_fig, rt_fig, data_file, large_cities):
    """Optimize and save figures and data for further processing.
    """

    Julia.eval("""
        optimize!(m)
        pre_rt = value.(m[:rt]); i = value.(m[:i])
        rt = expand(pre_rt, prm)
    """)

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
    basic_prm, cities_data, mob_matrix, target = read_data(options)
    ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
    force_dif = np.ones((ncities, ndays))
    prepare_optimization(basic_prm, cities_data, mob_matrix, target, force_dif)
    optimize_and_show_results("results/cmd_i_new.png", "results/cmd_rt_new.png",
                              "results/cmd_new.csv", cities_data.index)

if __name__ == "__main__":
    main()
