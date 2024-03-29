"""
Robot dance

Implements an automatic control framework to design efficient mitigation strategies
for Covid-19 based on the control of a SEIR model.

Copyright: Paulo J. S. Silva <pjssilva@unicamp.br>, 2020.
"""

using JuMP
using Ipopt
using Printf
using LinearAlgebra
using SparseArrays
using Distributions
import Statistics.mean
include("simple_arts.jl")

"""
    struct SEIR_Parameters

Parameters to define a SEIR model:

- tinc: incubation time (5.2 for Covid-19).
- tinf: time of infection (2.9 for Covid-19).
- rep: The natural reproduction rate of the disease, for examploe for Covid-19
  you might want to use 2.5 or a similar value.
- ndays: simulation duration.
- ncities: number of (interconnected) cities in the model.
- s1, e1, i1, r1: start proportion of the population in each SEIR class.
- alternate: the weight to give to alternation in the solution.
- availICU: vector with the ratio of avaible ICU.
- time_icu: mean stay in ICU
- rho_icu_ts: information on the time series to estimate the ratio of ICU needed
    for each infected.
- window: time window to keep rt constant.
- out: vector that represents the proportion of the population that leave each city during
    the day.
- M: Matrix that has at position (i, j) how much population goes from city i to city j, the
    proportion with respect to the population of the destiny (j). It should have 0 on the
    diagonal.
- Mt: Matrix that has at position (i, j) how much population goes from city j to city i, the
    proportion with respect to the population of the origin (j). It should have 0 on the
    diagonal.
"""
struct SEIR_Parameters
    # Basic epidemiological constants that define the SEIR model
    tinc::Float64
    tinf::Float64
    rep::Float64
    ndays::Int64
    ncities::Int64
    s1::Vector{Float64}
    e1::Vector{Float64}
    i1::Vector{Float64}
    r1::Vector{Float64}
    alternate::Float64
    availICU::Vector{Float64}
    time_icu::Int64
    rho_icu_ts::Matrix{Float64}
    window::Int64
    out::Vector{Float64}
    M::SparseMatrixCSC{Float64,Int64}
    Mt::SparseMatrixCSC{Float64,Int64}
    sars_over_cov::Float64
    sick_tested::Float64

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate,
        availICU, rho_icu_ts, window, out, M, Mt)

    SEIR parameters with mobility information (out, M, Mt).
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
        time_icu, rho_icu_ts, window, out, M, Mt)
        ls1 = length(s1)
        @assert length(e1) == ls1
        @assert length(i1) == ls1
        @assert length(r1) == ls1
        @assert length(availICU) == ls1
        @assert time_icu > 0.0
        @assert size(rho_icu_ts) == (ls1, 10)
        @assert size(out) == (ls1,)
        @assert all(out .>= 0.0)
        @assert size(M) == (ls1, ls1)
        @assert all(M .>= 0.0)
        @assert size(Mt) == (ls1, ls1)
        @assert all(Mt .>= 0.0)

        # For now sars_over_cov and sick_tested are hard coded
        sars_over_cov, sick_tested = 4.0, 0.4
        new(tinc, tinf, rep, ndays, ls1, s1, e1, i1, r1, alternate, availICU, time_icu,
            rho_icu_ts, window, out, M, Mt, sars_over_cov, sick_tested)
    end

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate,
        availICU, rho_icu_ts, window)

    SEIR parameters without mobility information, which is assumed to be 0.
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
        time_icu, rho_icu_ts, window)
        ls1 = length(s1)
        out = zeros(ls1)
        M = spzeros(ls1, ls1)
        Mt = spzeros(ls1, ls1)
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
            time_icu, rho_icu_ts, window, out, M, Mt)
    end

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU)

    SEIR parameters with unit time window and without mobility information, which is assumed 
    to be 0.
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU,
        time_icu, rho_icu_ts)
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
            time_icu, rho_icu_ts, 1)
    end
end


"""
    mapind(d, prm)

    Allows for natural use of rt while computing the right index mapping in time d.
"""
function mapind(d, prm)
    return prm.window*div(d - 1, prm.window) + 1
end


"""
    expand(rt, prm)

Expand rt to a full prm.ndays vector.
"""
function expand(rt, prm)
    full_rt = zeros(prm.ncities, prm.ndays)
    for c in 1:prm.ncities, d in 1:prm.ndays
        full_rt[c, d] = rt[c, mapind(d, prm)]
    end
    return full_rt
end


"""
    best_linear_solver

    Helper function to check what is the best linear solver available for Ipopt, 
        ma97 (preferred) or mumps.
"""
function best_linear_solver()
    PREFERRED = "ma97"
    #PREFERRED = "pardiso"
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
              "print_level" => 0, "linear_solver" => PREFERRED))
    @variable(m, x)
    @objective(m, Min, x^2)
    optimize!(m)
    if termination_status(m) != MOI.INVALID_OPTION
        return PREFERRED
    else
        return "mumps"
    end
end


"""
    seir_model_with_free_initial_value(prm)

Build an optimization model with the SEIR discretization as constraints. The inicial
parameters are not initialized and remain free. This can be useful, for example, to fit the
initial parameters to observed data.
"""
function seir_model_with_free_initial_values(prm, verbosity=0, tau=3, test_efficacy=0.8)
    # Save work and get col indices of both M and Mt
    coli_M = [findnz(prm.M[:,c])[1] for c in 1:prm.ncities]
    coli_Mt = [findnz(prm.Mt[:,c])[1] for c in 1:prm.ncities]

    # Create the optimization model.
    # I am reverting to mumps because I can not limit ma97 to use
    # only the actual cores in my machine and mumps seems to be doing 
    # fine.
    if verbosity >= 1
        println("Initializing optimization model...")
    end

    verbosity_ipopt = 0
    if verbosity >= 1
        verbosity_ipopt = 5 # Print summary and progress
    end

    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => verbosity_ipopt, "linear_solver" => best_linear_solver()))
    if verbosity >= 1
        println("Initializing optimization model... Ok!")
    end
    # For simplicity I am assuming that one step per day is OK.
    dt = 1.0

    # Note that I do not fix the initial state. It should be defined elsewhere.
    # State variables
    if verbosity >= 1
        println("Adding variables to the model...")
    end
    @variable(m, 0.0 <= s[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= e[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= i[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= q[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= r[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= test[1:prm.ncities, 1:prm.ndays] <= 1.0)

    # Constants that define the testing impact
    test_const = 1/prm.sars_over_cov*test_efficacy*exp(-tau/prm.tinf)

    # Control variable
    @variable(m, 0.0 <= rt[1:prm.ncities, 1:prm.window:prm.ndays] <= prm.rep)
    
    # Extra variables to better separate linear and nonlinear expressions and
    # to decouple and "sparsify" the matrices.
    # Obs. I tried many variations, only adding the variable below worded the best.
    #      I tried to get rid of all SEIR variables and use only the initial conditions.
    #      Add variables for sp, ep, ip, rp. Add a variable to represent s times i.
    @variable(m, p_eff_p_c[1:prm.ncities, 1:prm.ndays])
    @variable(m, i_eff_p_c[1:prm.ncities, 1:prm.ndays])
    @variable(m, i_eff[1:prm.ncities, t=1:prm.ndays])
    @variable(m, rti_eff[1:prm.ncities, t=1:prm.ndays])
    @variable(m, rti[1:prm.ncities, t=1:prm.ndays])
    if verbosity >= 1
        println("Adding variables to the model... Ok!")
    end

    # Expressions that define "sub-states"

    # Parameter that determines the proportion of I that can not travel.
    FIXED_I = 0.0
    CAN_TRAVEL_I = 1.0 - FIXED_I

    if verbosity >= 1
        println("Defining additional expressions...")
    end

    if FIXED_I > 0
        @expression(m, can_travel[c=1:prm.ncities, t=1:prm.ndays],
            1.0 - FIXED_I*i[c, t] - q[c, t]
        )
    else
        @expression(m, can_travel[c=1:prm.ncities, t=1:prm.ndays], 
            1.0 - q[c, t]
        )
    end

    @expression(m, alt_out[c=1:prm.ncities, d=1:prm.window:prm.ndays],
        sum(rt[k, d]/prm.rep*prm.Mt[k, c] for k in coli_Mt[c])
    )
    @expression(m, dest_orig[c=1:prm.ncities, k in coli_M[c], d=1:prm.window:prm.ndays],
        rt[c, d]/prm.rep*prm.M[k, c]
    )
    @expression(m, orig_dest[c=1:prm.ncities, k in coli_Mt[c], d=1:prm.window:prm.ndays],
        rt[k, d]/prm.rep*prm.Mt[k, c]
    )

    # p_eff_p_c denotes the proportion of the effective population at city c
    # during the day divided by the original population of city c

    @constraint(m, [c=1:prm.ncities, t=1:prm.ndays],
        p_eff_p_c[c, t] == 1.0 - alt_out[c, mapind(t, prm)]*can_travel[c, t] + 
            sum(dest_orig[c, k, mapind(t, prm)]*can_travel[k, t] for k in coli_M[c])
    )
    # i_eff_p_c denotes the proportion of the effective number of infected at city c
    # during the day divided by the original population of city c
    @constraint(m, [c=1:prm.ncities, t=1:prm.ndays],
        i_eff_p_c[c, t] == (1.0 - alt_out[c, mapind(t, prm)]*CAN_TRAVEL_I)*i[c, t] +
            sum(dest_orig[c, k, mapind(t, prm)]*CAN_TRAVEL_I*i[k, t] for k in coli_M[c])
    )
    # i_eff is the effective ratio of inffected in city c
    @constraint(m, [c=1:prm.ncities, t=1:prm.ndays], 
        i_eff[c, t]*p_eff_p_c[c, t] == i_eff_p_c[c, t]
    )

    # rti is rt at city c times the i_eff at city c
    @constraint(m, [c=1:prm.ncities, t=1:prm.ndays],
        rti_eff[c, t] == rt[c, mapind(t, prm)]*i_eff[c, t]
    )

    @constraint(m, [c=1:prm.ncities, t=1:prm.ndays],
        rti[c, t] == rt[c, mapind(t, prm)]*i[c, t]
    )
    
    # Parameter that measures how much important is the infection during the day
    # when compared to the night.
    α = 2/3

    # Compute the gradients at time t of the SEIR model.

    # Estimates the infection rate of the susceptible people from city c
    # that went to the other cities k.
    if verbosity >= 1
        println("Defining SEIR equations...")
    end
    @variable(m, one_minus_out_s[c=1:prm.ncities, t=1:prm.ndays])
    @constraint(m, [c=1:prm.ncities, t=1:prm.ndays],
        one_minus_out_s[c, t] == (1.0 - alt_out[c, mapind(t, prm)])*s[c, t]
    )
    @variable(m, orig_dest_s[c=1:prm.ncities, k in coli_Mt[c], t=1:prm.ndays])
    @constraint(m, [c=1:prm.ncities, k in coli_Mt[c], t=1:prm.ndays],
        orig_dest_s[c, k, t] == orig_dest[c, k, mapind(t, prm)]*s[c, t]
    )
    @expression(m, ds[c=1:prm.ncities, t=1:prm.ndays],
        -1.0/prm.tinf*(
        α*( one_minus_out_s[c,t]*rti_eff[c, t] +
            sum(orig_dest_s[c, k, t]*rti_eff[k, t] for k = coli_Mt[c]) ) 
        + (1 - α)*s[c, t]*rti[c, t]
       )
    )
    @expression(m, de[c=1:prm.ncities, t=1:prm.ndays],
        -ds[c, t] - (1.0/prm.tinc)*e[c,t]
    )
    @expression(m, di[c=1:prm.ncities, t=1:prm.ndays],
        (1.0/prm.tinc)*e[c, t] - (1.0/prm.tinf)*(i[c, t] + test_const*test[c, t])
    )
    # TODO: add a Tq in prm?
    tq = 3*prm.tinf
    @expression(m, dq[c=1:prm.ncities, t=1:prm.ndays],
        test_const/prm.tinf*test[c, t] - 1.0/tq*q[c, t]
    )
    @expression(m, dr[c=1:prm.ncities, t=1:prm.ndays],
        1/tq*q[c, t] + (1.0/prm.tinf)*i[c, t]
    )
    if verbosity >= 1
        println("Defining SEIR equations... Ok!")
    end

    discr_method = "finite_difference"
    k_curr_t = 0.5
    k_prev_t = 0.5

    # Discretize SEIR equations
    if discr_method == "finite_difference"
        if verbosity >= 1
            if k_curr_t == 1.0 && k_prev_t == 0.0
                println("Discretizing SEIR equations (backward)...")
            elseif k_curr_t == 0.0 && k_prev_t == 1.0
                println("Discretizing SEIR equations (forward)...")
            elseif k_curr_t == 0.5 && k_prev_t == 0.5
                println("Discretizing SEIR equations (central)...")
            end
        end

        @constraint(m, [c=1:prm.ncities, t=2:prm.ndays],
            s[c, t] == s[c, t - 1] + (k_prev_t*ds[c, t-1] + k_curr_t*ds[c, t])*dt
        )
        @constraint(m, [c=1:prm.ncities, t=2:prm.ndays],
            e[c, t] == e[c, t - 1] + (k_prev_t*de[c, t-1] + k_curr_t*de[c, t])*dt
        )
        @constraint(m, [c=1:prm.ncities, t=2:prm.ndays],
            i[c, t] == i[c, t - 1] + (k_prev_t*di[c, t-1] + k_curr_t*di[c, t])*dt
        )
        @constraint(m, [c=1:prm.ncities, t=2:prm.ndays],
            q[c, t] == q[c, t - 1] + (k_prev_t*dq[c, t-1] + k_curr_t*dq[c, t])*dt
        )
        @constraint(m, [c=1:prm.ncities, t=2:prm.ndays],
            r[c, t] == r[c, t - 1] + (k_prev_t*dr[c, t-1] + k_curr_t*dr[c, t])*dt
        )
        if verbosity >= 1
            println("Discretizing SEIR equations... Ok!")
        end
    else
        throw("Invalid discretization method")
    end

    return m
end


"""
    seir_model(prm)

Creates a SEIR model setting the initial parameters for the SEIR variables from prm.
"""
function seir_model(prm, verbosity, tau=3, test_efficacy=0.8)
    m = seir_model_with_free_initial_values(prm, verbosity, tau, test_efficacy)

    # Initial state
    s1, e1, i1, q1, r1 = m[:s][:, 1], m[:e][:, 1], m[:i][:, 1], m[:q][:, 1], m[:r][:, 1]
    for c in 1:prm.ncities
        fix(s1[c], prm.s1[c]; force=true)
        fix(e1[c], prm.e1[c]; force=true)
        fix(i1[c], prm.i1[c]; force=true)
        fix(r1[c], prm.r1[c]; force=true)
        fix(q1[c], 
            max(0.0, 1.0 - prm.s1[c] - prm.e1[c] - prm.i1[c] - prm.r1[c]); force=true)
    end
    return m
end


"""
    fixed_rt_model(prm)

Creates a model from prm setting all initial parameters and defining the RT as the natural
R0 set in prm.
"""
function fixed_rt_model(prm)
    m = seir_model(prm)
    rt = m[:rt]

    # Fix all rts
    for c = 1:prm.ncities
        for t = 1:prm.window:prm.ndays
            fix(rt[c, t], prm.rep; force=true)
        end
    end
    return m
end


"""
    fit_initcond_model(tinc, tinf, rep, initial_data, ttv_weight=0.25)

Fit the initial parameters for a single city using squared relative error and
allowing rt to change to capture social distancing measures implemented in the
mean time.

# Arguments

- ttv_weight: controls the wight given to the total variation of the R0 parameter.
"""
function fit_initial(tinc, tinf, rep, time_icu, data, ttv_weight=0.25)
    # Create SEIR model
    prm = SEIR_Parameters(tinc, tinf, rep, length(data), [1.0], [0.0], [0.0], [0.0], 
        0.0, [0.0], time_icu, zeros(1, 10), 1, [0.0], zeros(1, 1), zeros(1, 1))

    m = seir_model_with_free_initial_values(prm)

    # Initial state
    s1, e1, i, q, r1, rt, test = m[:s][1, 1], m[:e][1, 1], m[:i], m[:q], m[:r][1, 1], m[:rt], m[:test]
    set_start_value(r1, prm.r1[1])
    set_start_value(s1, prm.s1[1])
    set_start_value(e1, prm.e1[1])
    set_start_value(i[1, 1], prm.i1[1])
    for c =1:prm.ncities, d=1:prm.ndays
        fix(test[c, d], 0.0; force=true)
        fix(q[c, d], 0.0; force=true)
    end

    # Define upper bounds on rt
    for t = 1:prm.window:prm.ndays
        set_upper_bound(rt[1, t], 10*prm.rep)
    end

    # USed to compute the total rt variation
    @variable(m, ttv[2:prm.ndays])
    @constraint(m, con_ttv[i=2:prm.ndays], ttv[i] >= rt[1, i - 1] - rt[1, i])
    @constraint(m, con_ttv2[i=2:prm.ndays], ttv[i] >= rt[1, i] - rt[1, i - 1])

    # SEIR contraint
    @constraint(m, s1 + e1 + i[1, 1] + r1 == 1.0)

    # Define objective function
    # Compute a scaling factor so as a least square objective makes more sense
    valid = (t for t = 1:prm.ndays if data[t] > 0)
    @NLobjective(m, Min, sum((i[1, t]/data[t] - 1.0)^2 for t in valid) + ttv_weight*sum(ttv[t] for t = 2:prm.ndays))

    # Optimize
    optimize!(m)

    return value(m[:s][1, prm.ndays]), value(m[:e][1, prm.ndays]), 
        value(m[:i][1, prm.ndays]), value(m[:r][1, prm.ndays]), value.(rt[1, :])
end


"""
    window_control_multcities

Built a simple control problem that tries to force the infected to remain below target every
day for every city using daily controls but only allow them to change in the start of the
time windows.

# Arguments

- prm: SEIR parameters with initial state and other informations.
- population: population of each city.
- target: limit of infected to impose at each city for each day.
- window: number of days for the time window.
- force_difference: allow to turn off the alternation for a city in certain days. Should be
    used if alternation happens even after the eipdemy is dieing off.
- hammer_durarion: Duration in days of a intial hammer phase.
- hammer: Rt level that should be achieved during hammer phase.
- min_rt: minimum rt achievable outside the hammer phases.
- pools: groups of regions to be treated as a pool in the ICU constraints.
"""
function window_control_multcities(prm, population, target, force_difference, 
    hammer_duration=0, hammer=0.89, min_rt=1.0, pools=[[c] for c in 1:prm.ncities],
    verbosity=0, test_budget=0, tests_off=Int64[], 
    tau=3, test_efficacy=0.8, daily_tests=0, proportional_test=false)
    @assert sum(mod.(hammer_duration, prm.window)) == 0

    # TODO: These should be parameters - first try
    times_series = [Simple_ARTS(prm.rho_icu_ts[c, :]...) for c in 1:prm.ncities]

    m = seir_model(prm, verbosity, tau, test_efficacy)

    if verbosity >= 1
        println("Setting limits for rt...")
    end
    # Fix rt during hammer phase
    rt = m[:rt]
    for c = 1:prm.ncities, d = 1:prm.window:hammer_duration[c]
        fix(rt[c, d], hammer[c]; force=true)
    end
    
    # Set the minimum rt achievable after the hammer phase.
    for c = 1:prm.ncities, d = hammer_duration[c] + 1:prm.window:prm.ndays
        set_lower_bound(rt[c, d], min_rt)
    end
    if verbosity >= 1
        println("Setting limits for rt... Ok!")
    end

    # Bound the maximal infection rate taking into account the maximal ICU rooms available
    # using a chance contraint.
    if verbosity >= 1
        println("Setting limits for number of infected...")
    end

    # We implement two variants one based on max I and another on sum on
    # entering in R, the max I is simples and gives good results so we
    # are keeping it in the code.
    i = m[:i]
    firstday = hammer_duration .+ 1

    println(pools)
    n_pools = length(pools)
    first_pool_day = [minimum(firstday[pool]) for pool in pools]    
    pool_population = [sum(population[pool]) for pool in pools]
    @variable(m, V[p=1:n_pools, d=first_pool_day[p]:prm.ndays - prm.time_icu] >= 0)
    for p in 1:n_pools
        pool = pools[p]
        # Uses the time series that gives the ratio of IC needed from the first
        # city in the pool.
        ρ_icu = times_series[pool[1]]
        reset(ρ_icu)

        # As in the paper, V represents the number of people that will leave
        # infected and potentially go to ICU.
        @constraint(m, [d=first_pool_day[p]:prm.ndays - prm.time_icu],
            V[p, d] == prm.time_icu/prm.tinf*sum(population[c]*i[c, d] for c in pool) / pool_population[p]
        )

        for d in 1:prm.ndays - prm.time_icu
            Eicu, safety_level = iterate(ρ_icu)
            if d >= first_pool_day[p]
                @constraint(m,
                    (Eicu + safety_level)*V[p, d] <= 
                    sum(target[c, d]*population[c]*prm.availICU[c] for c in pool) / pool_population[p]
                )
            end
        end
    end

    # Constraints on the tests
    if test_budget > 0
        test, i = m[:test], m[:i]
        max_pop = maximum(population)
        # Only use the given budget of tests
        @expression(m, total_tests[d=1:prm.ndays],
            sum(population[c]*test[c, d] for c = 1:prm.ncities) 
        )
        @constraint(m, use_test_available,
            sum(total_tests)/max_pop <= test_budget/max_pop
        )
        # Maximal ammount of daily test
        @constraint(m, max_day[d=1:prm.ndays],
            total_tests[d]/max_pop <= daily_tests/max_pop
        )

        @constraint(m, test_only_present[c=1:prm.ncities, d=1:prm.ndays],
            test[c, d] <= prm.sars_over_cov * prm.sick_tested * i[c, d] / prm.tinf
        )

        # Disallow test in some regions.
        for c in tests_off
            for d = 1:prm.ndays
                fix(test[c, d], 0.0; force=true)
            end
        end 

        # Limit tests to the proportion of infected in the privious day
        if proportional_test
            @expression(m, total_infected[d=1:prm.ndays],
                sum(i[c, d]*population[c] for c = 1:prm.ncities)
            )
            @constraint(m, proprional_tests[c=1:prm.ncities, d=2:prm.ndays],
                test[c, d]*total_infected[d - 1]/max_pop <= i[c, d - 1]*total_tests[d]/max_pop
            )
        end

    else
        test = m[:test]
        for c = 1:prm.ncities, d = 1:prm.ndays
            fix(test[c, d], 0.0; force=true)
        end
    end
    

    if verbosity >= 1
        println("Setting limits for number of infected... Ok!")
    end

    # Compute the weights for the objectives terms
    if verbosity >= 1
        println("Computing objective function...")
    end
    effect_pop = population # You may try to use other metrics like sqrt.(population)
    mean_population = mean(effect_pop)
    dif_matrix = Matrix{Float64}(undef, prm.ncities, prm.ndays)
    for c = 1:prm.ncities, d = 1:prm.ndays
        dif_matrix[c, d] = force_difference[c, d] / mean_population / (2*prm.ncities)
    end
    # Define objective
    @objective(m, Min,
        # Try to keep as many people working as possible
        prm.window*sum(effect_pop[c]/mean_population*(prm.rep - rt[c, d])
            for c = 1:prm.ncities for d = hammer_duration[c]+1:prm.window:prm.ndays) -
        # Induce bang-bang
        0.02*prm.alternate*prm.window*sum(
            force_difference[c, d]*effect_pop[c]/mean_population*
            (prm.rep - rt[c, d])*(min_rt - rt[c, d])
            for c = 1:prm.ncities for d = hammer_duration[c]+1:prm.window:prm.ndays) -
        # Try to alternate within a single city.
        0.5*prm.alternate*prm.window/(prm.rep^2)*sum(
            force_difference[c, d]*(rt[c, d] - rt[c, d - prm.window])^2 
            for c = 1:prm.ncities 
            for d = hammer_duration[c] + prm.window + 1:prm.window:prm.ndays
        ) -
        # Try to enforce different cities to alternate the controls
        0.5*prm.alternate*prm.window/(prm.rep^2)*sum(
            minimum((effect_pop[c], effect_pop[cl]))*
            minimum((dif_matrix[c, d], dif_matrix[cl, d]))*
            (rt[c, d] - rt[cl, d])^2
            for c = 1:prm.ncities 
            for cl = c + 1:prm.ncities 
            for d = hammer_duration[c] + 1:prm.window:prm.ndays
        )
    )
    if verbosity >= 1
        println("Computing objective function... Ok!")
    end

    return m
end


"""
    add_ramp

Adds ramp constraints to the robot-dance model.

# Input arguments
m: the optimization model
prm: struct with parameters
hammer_duration: initial hammer for each city
delta_rt_max: max increase in the control rt between (t-1) and (t)

# Output: the optimization model m
"""
function add_ramp(m, prm, hammer_duration, delta_rt_max, verbosity=0)
    if verbosity >= 1
        println("Adding ramp constraints (delta_rt_max = $delta_rt_max)...")
    end
    rt = m[:rt]
    @constraint(m, [c=1:prm.ncities, d = hammer_duration[c] + 1:prm.window:prm.ndays],
    rt[c, d] - rt[c, d - prm.window] <= delta_rt_max
    )
    if verbosity >= 1
        println("Adding ramp constraints (delta_rt_max = $delta_rt_max)... Ok!")
    end

    return m
end


"""
    simulate_control

Simulate the SEIR mdel with a given control checking whether the target is achieved.

TODO: fix this, hammer_duration does not even exist.

"""
function simulate_control(prm, population, control, target)
    @assert sum(mod.(hammer_duration[c], prm.window)) == 0

    m = seir_model(prm)

    rt = m[:rt]
    for c=1:prm.ncities, d=1:prm.ndays
        fix(rt[c, mapind(d, prm)], control[c, d]; force=true)
    end

    # Constraint on maximum level of infection
    i = m[:i]
    @constraint(m, [c=1:prm.ncities, t=2:prm.ndays], i[c, t] <= target)

    # Compute the weights for the objectives terms
    effect_pop = sqrt.(population)
    mean_population = mean(effect_pop)
    dif_matrix = Matrix{Float64}(undef, prm.ncities, prm.ndays)
    for c = 1:prm.ncities, d = 1:prm.ndays
        dif_matrix[c, d] = force_difference[c, d] / mean_population / (2*prm.ncities)
    end
    # Define objective
    @objective(m, Min,
        # Try to keep as many people working as possible
        sum(prm.window*effect_pop[c]/mean_population*(prm.rep - rt[c, d])
            for c = 1:prm.ncities for d = +1:prm.window:prm.ndays) -
        # Try to enforce different cities to alternate the controls
        100.0/(prm.rep^2)*sum(
            minimum((effect_pop[c], effect_pop[cl]))*
            minimum((dif_matrix[c, d], dif_matrix[cl, d]))*
            (rt[c, d] - rt[cl, d]^2)
            for c = 1:prm.ncities 
            for cl = c + 1:prm.ncities 
            for d = hammer_duration[c] + 1:prm.window:prm.ndays
            )
    )

    return m
end
