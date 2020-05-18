"""
Helper functions to run tests.
"""

import sys
import seir
import pandas as pd
import numpy as np
import numpy.linalg as la
import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 7

# To use PyJulia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main as Julia
Julia.eval('include("robot_dance.jl")')

# Configuration
# Warning: the two constants below have to be the same used in the optimization code.
# Configuration
TINC = 2.9
TINF = 5.2


def initial_conditions(city, covid_data, covid_window, min_days, Julia, correction=1.0):
    """Fits data and define initial contidions of the SEIR model.
    """
    # Gets the city data
    city_data = covid_data[covid_data["city"] == city].copy()
    city_data.reset_index(inplace=True)
    city_data.sort_values(by=["date"], inplace=True)
    population = city_data["estimated_population_2019"].iloc[0]
    confirmed = city_data["confirmed"]

    # I am computing the new cases instead of using the new_confirmed column because
    # there is error at least in the first element for São Paulo. It should be 1.
    new_cases = confirmed.values[1:] - confirmed.values[:-1]
    new_cases = np.append(confirmed[0], new_cases)
    city_data["new_cases"] = new_cases
    
    observed_I = city_data["new_cases"].rolling(covid_window).sum()
    observed_I[:covid_window] = confirmed[:covid_window]
    ndays = len(observed_I)
    if ndays >= min_days:
        observed_I /= population
        Julia.observed_I = correction*observed_I.values
        Julia.eval('initialc = fit_initial(observed_I)')
        S0 = Julia.initialc[0]
        E0 = Julia.initialc[1]
        I0 = Julia.initialc[2]
        R0 = Julia.initialc[3]
        return (S0, E0, I0, R0, ndays), observed_I
    else:
        raise ValueError("Not enough data for %s only %d days available" % 
            (city, len(observed_I)))


def simulate(parameters, c, covid_data, covid_window, min_days):
    """Simulate from the computed initial parameters until the last day.
    """
    # Get the city data to find out the last day (this is overkill, I know)
    city_data = covid_data[covid_data["city"] == c].copy()
    city_data.reset_index(inplace=True)
    city_data.sort_values(by=["date"], inplace=False)

    S0, E0, I0, R0, ndays = parameters[c]
    covid = seir.seir(ndays)
    last_day = city_data["date"].iloc[-1]
    print("Simulating", c, "until", last_day)
    result = covid.run((S0,E0,I0,R0))
    return result[:, -1], last_day


def compute_initial_condition_and_evolve(correction):
    # Read data and define what are the cities of interest
    covid_data = pd.read_csv("data/covid_with_cities.csv")
    covid_data = covid_data[covid_data["state"] == "SP"]
    large_cities = \
        covid_data[covid_data["estimated_population_2019"] > 100000]["city"].unique()

    # Compute initial parameters fitting the data
    covid_window = int(round(TINC*TINF))
    min_days = 5    
    parameters = {}
    ignored = []

    n_cities = len(large_cities)
    for i in range(n_cities):
        c = large_cities[i]
        print("%d/%d" %(i + 1, n_cities), c)
        try:
            parameters[c], observed_I = initial_conditions(c, covid_data, covid_window, 
                min_days, Julia, correction)
        except ValueError:
            print("Ignoring ", c, "not enough data.")
            ignored.append(c)    

    # Simulate the data until the last day to start the optimization phase.
    parameters_at_final_day = {}
    for c in large_cities:
        parameters_at_final_day[c], last_day = simulate(parameters, c, covid_data,
            covid_window, min_days)

    # Save results
    parameters_at_final_day = pd.DataFrame.from_dict(parameters_at_final_day, 
        orient="index", columns=["S0", "E0", "I0", "R0"])
    with open('data/initial_values.csv', 'w') as f:
        f.write("# Initial condition for " + str(last_day) + "\n")
        parameters_at_final_day.to_csv(f)


def read_test_data(max_neighbors):
    """Read the data used in a robot dance simulation.

       max_neighbors: maximum number of neighbors allowed in the mobility matrix.
    """
    # Read data and define what are the cities of interest
    covid_data = pd.read_csv(
        "/home/pjssilva/nuvem/unicamp/compartilhados/ICMCxCovid/Data/covid_with_cities.csv")
    covid_data = covid_data[covid_data["state"] == "SP"]
    large_cities = [i.upper() for i in 
        covid_data[covid_data["estimated_population_2019"] > 500000]["city"].unique()
    ]
    if "ARAÇATUBA" not in large_cities:
        large_cities.append("ARAÇATUBA")
    if "SÃO JOSÉ DO RIO PRETO" not in large_cities:
        large_cities.append("SÃO JOSÉ DO RIO PRETO")
    large_cities.sort()

    # Read the mobility_matrix
    mobility_matrix = pd.read_csv("data/move_mat_SÃO PAULO_SP-Municipios_norm.csv",
        header=None, sep=" ")
    city_names = pd.read_csv("data/move_mat_SÃO PAULO_SP-Municipios_reg_names.txt", 
        header=None)

    # Cut the matrix to see only the desired cities
    mobility_matrix.index = city_names[0]
    mobility_matrix.columns = city_names[0]
    mobility_matrix = mobility_matrix.loc[large_cities, large_cities].T
    mobility_matrix = mobility_matrix.mask(
        mobility_matrix.rank(axis=1, method='min', ascending=False) > max_neighbors + 1, 0
    )

    # Read the initial values
    initial_values = pd.read_csv("data/initial_values.csv", header=1, index_col=0)
    # Get the estimated population. This is a little brute force, but...
    covid_data = pd.read_csv(
        "/home/pjssilva/nuvem/unicamp/compartilhados/ICMCxCovid/Data/covid_with_cities.csv")
    population = {}

    for c in initial_values.index:
        population[c] = covid_data[covid_data["city"] == c]["estimated_population_2019"].iloc[0]

    population = pd.DataFrame.from_dict(population, orient="index", columns=["Population"])
    initial_values.index = [i.upper() for i in initial_values.index]
    population.index = [i.upper() for i in population.index]
    population = population.loc[large_cities]
    initial_values = initial_values.loc[large_cities]

    # Adjust the mobility matrix
    np.fill_diagonal(mobility_matrix.values, 0.0)
    # out vector has at entry i the proportion of the population of city i that leaves the
    # city during the day
    out = mobility_matrix.sum(axis = 1)

    # The M matrix has at entry [i, j] the proportion, with respect to the population of j, 
    # of people from i that spend the day in j
    M = mobility_matrix.copy()
    for i in mobility_matrix.index:
        M.loc[i] = (mobility_matrix.loc[i] * population.loc[i].values / 
                    population["Population"])

    return large_cities, population, initial_values, M, out


def prepare_optimization(large_cities, population, initial_values, M, out,
    target, window=14, ndays=400, min_level=1.0, hammer_duration=14, hammer_level=0.89, 
    force_dif=1):
    # Infected upper bound, it is larger in São Paulo.
    ncities = len(large_cities)

    if force_dif is 1:
        force_dif = np.ones((ncities, ndays))

    Julia.s1 = initial_values.loc[large_cities, "S0"].values
    Julia.e1 = initial_values.loc[large_cities, "E0"].values
    Julia.i1 = initial_values.loc[large_cities, "I0"].values
    Julia.r1 = initial_values.loc[large_cities, "R0"].values
    Julia.out = out.values
    Julia.M = M.values.copy()
    Julia.population = population.values.copy()
    Julia.ndays = ndays
    Julia.target = target
    Julia.hammer_duration = hammer_duration
    Julia.hammer_level = hammer_level
    Julia.min_level = min_level
    Julia.force_dif = force_dif
    if window == 1:
        Julia.eval("""
            prm = SEIR_Parameters(ndays, s1, e1, i1, r1, out, sparse(M), sparse(M'))
            m = control_multcities(prm, population, target, force_dif, hammer_duration, 
                                   hammer_level, min_level)
        """)
    else:
        Julia.window = window
        Julia.eval("""
            prm = SEIR_Parameters(ndays, s1, e1, i1, r1, out, sparse(M), sparse(M'))
            m = window_control_multcities(prm, population, target, window, force_dif, 
                                          hammer_duration, hammer_level, min_level);
        """);        


def save_result(cities_names, filename):
    """Save the result of a run for further processing.
    """
    Julia.eval("s = value.(m[:s]); e = value.(m[:e]); i = value.(m[:i]); r = value.(m[:r])")
    Julia.eval("rt = value.(m[:rt])")
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
        rt = value.(m[:rt]); i = value.(m[:i])
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

