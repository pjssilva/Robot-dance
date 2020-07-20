# Robot Dance

This repository contains the code used in the manuscript 
[Robot dance: a city-wise automatic control of Covid-19 mitigation levels](https://www.medrxiv.org/content/10.1101/2020.05.11.20098541v1).

The idea is to sue control and optimization to design efficient alternate mitigation
strategies for Covid-19 and other infectious diseases. It uses an adapted SEIR model that
takes into account the mobility between different cities. The objective is to design a
protocols that controls the number of infected to a maximum level while alternating between
more stringent and more light mitigation and between cities (or regions), so that the State
or country does not close all at once.

The code up to now should work for a few cities (or regions), a couple of tens. We have
some ideas to make it scale to the hundreds.

## Installing dependencies

This code is written in [Julia](https://www.julialang.org) and
[Python](https://www.python.org), so you need both installed. To install Julia you should
go to its website, select download and follow instructions. After installing Julia you need
to use its package manager and install the `JuMP` and `Ipopt` packages. 

To get good performance out of Ipopt, it is very important to use a high performance linear
solver. The default free solver, `mumps` works well for 10 cities or so. If you want to
solve larger problems with tens of cities you will need to compile Ipopt with support to
[HSL](http://www.hsl.rl.ac.uk) linear solvers. Robot dance uses MA97 in the code whenever
available. I have also prepared a little [document](compiling_ipopt.md) describing how to
compile Ipopt with HSL under Ubuntu.

There are many ways to install Python. I use
[Anaconda](https://www.anaconda.com/products/individual). By installing Anaconda you get a
comprehensive and updated Python environment with packages the code use like `numpy`,
`scipy` , and `matplotlib`.

After that install [PyJulia](https://github.com/JuliaPy/pyjulia).

## Running and input files

To run the basic model described in the report on a problem instance run the Python code
`run_robot.py`. This code expects the input files to be located in the `data` sub-directory.
It expects 4 input files in the subfolder `data`:

* `basic_paramters.csv`: simple CSV file with basic parameters, one per line in the format
  `<parameter name>, <paramter values>`. The parameters are:
    * `tinc`: Incubation time for the SEIR Model (default 5.2).
    * `tinf`: Infected time for SEIR model (default 2.9).
    * `rep`: reproduction rate of the virus without mitigation (default 2.5)
    * `ndays`: number of days to be considered in the simulation.
    * `window`: number of days to keep the mitigation level constant.
    * `min_level`: what is the r0 attainable using the most demanding mitigation.
    * `delta_rt_max`: (optional) if this parameter is provided, ramp constraints will be added to the model, preventing rt to increase more than `delta_rt_max` between two consecutive periods (`delta_rt_max > 0`).

* `cities_data.csv`: basic cities data in CSV format. It must contain one line per city,
  with the city name as index and the following named columns:
    * `S1`: initial value for the S variable in the SEIR model.
    * `E1`: initial value for the E variable in the SEIR model.
    * `I1`: initial value for the I variable in the SEIR model.
    * `R1`: initial value for the R variable in the SEIR model.
    * `population`: population of the city. 
    
  Alternatively to this file, the user can make available the information with the
  (accumulated) number of infected to estimate the initial data `S1, E1, I1, R1`. This file
  should be named `pre_cities_data.csv` with columns labeled:
    * `city`: the city name. 
    * `state`: the state code if you want to select by state (or a neutral string).
    * `date`: a date where the accumulated infected was measured, they have to be made 
       daily from the start of the data for `city` up to a final common day. 
    * `confirmed`: the *accumulated* number of confirmed cases.
    * `estimated_population_2019`: contains the estimated population of the city (the
       same number repeated). 
  The file can have other columns, but only those will be used by the program.

* `mobility_matrix.csv`: the mobility matrix, with labels and index using the cities names.
  The cities must be the same cities that appear in `cities_data.csv` and in the same
  order. The position `(i, j)` of the matrix must contain the ratio of the population of
  city `i` that goes to city `j`, with the ratio computed with respect to the population of
  the target city `j`. The diagonal must be 0. Moreover, it there must be an extra, last,
  column labeled `out` that should contain the ration of the the total population of city
  at line `i` that leaves the city to work during the day. This ratio should be computed
  with respect to the population of `i`, the origin. If the file does not exist, the matrix
  is assumed to be 0 everywhere.

* `target.csv`: a matrix with the maximal amount of infected that is acceptable for each
  cities (as rows) at each day (as columns). The city names should be used as index and
  consecutive numbers from 1 to `ndays`as column labels. 

* `hammer_data.csv`: (optional) hammer to be applied in the first days/weeks, if necessary. If this file is provided, the algorithm will check if the hammer phase is long enough for each city; finding that it is not, it will increase the duration by one window and check again, until no city violate the target of number of infected after the hammer phase.
  It must contain one line per city, with the city name as index and the following named columns:
    * `duration`: minimum duration of hammer for the city, in days.
    * `level`: level (`r0`) to be applied during the hammer phase.


If this file is not provided, default data will be used: `duration = 0`, `level=0.89` and the iterative check above will be performed.

After you have all the files in place with the right names, you can run the code with
`python run_robot.py` the result will be made available in a file named
`results/cmd_res.csv`. It has the simulated values for the SEIR variables and the target
reproducible number at each day, labeled `rt`, for each city. You can also find two
pictures that help you quicly visualize the curves `results/cmd_i_res.png` and
`results/cmd_rt_new.png`.

TODO: Add files to describe how desirable is to alternate between any pair of cities (I am
using the minimum of the squared populations right now) and to turn off alternation after
the epidemic is controlled by herd immunity in a city, if necessary.

### Computational resources.

The code uses a highly parallel optimization solver to run a large scale optimization
problem, named Ipopt, which is installed as a Julia package. This demands a good computer.
A problem with more than 10-20 cities/regions the solver may face difficulties and stall.
In order to avoid this you may need to install Ipopt with HSL. HSL is free for research.
Please look at these [instructions of how to compile Ipopt](compiling_ipopt.md) if needed.
Also long time horizons (like the 400 days simulations we did in he report) are very
demanding. The code is not ready to run on a cluster. We will try to continuously improve
the code in order to overcome these limitations.

## Copyright 

Copyright Paulo J. S. Silva e Luis Gustavo Nonato. See the [license file](LICENSE.md).

## Funding

This research is supported by CeMEAI/FAPESP, Instituto Serrapilheira, and CNPq.

## Please Cite Us

We provide this code hoping that it will be useful for others. Please if you use it, let us
know about you experiences. Moreover, if you use this code in any publication, please cite
us. This is very important. For the moment we only have the manuscript, so cite it as

Silva, Paulo J. S., Pereira, Tiago and Nonato, Luis Gustavo. "Robot dance: a city-wise
automatic control of Covid-19 mitigation levels". Medrxiv, 2020.
[doi:https://doi.org/10.1101/2020.05.11.20098541]

