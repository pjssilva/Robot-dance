using JuMP
using Ipopt
using Printf
using Plots
using LinearAlgebra
using SparseArrays
import Statistics.mean

"""
    struct SEIR_Parameters

    Parameters to define a SEIR model:

    ndays: for the simulation duration.
    ncities: number of (interconnected) cities in the model.
    s1, e1, i1, r1: start proportion of the population in each SEIR class.
    out: vector that represents the proportion of the population that leaves
        each city during the day.
    M: Matrix that has at position (i, j) how much population goes from city
        i to city j, the proportion with respect to the population of the
        destiny (j).
    Mt: transpose of M.
    natural_rt: The natural reproduction rate of the disease, right now it
        is assumed constant related to Covid19 (2.5).
    tinc: incubation time related to Covid19 (5.2).
    tinf: time of infection related to Covid19 (2.9).
"""
struct SEIR_Parameters
    ndays::Int64
    ncities::Int64
    s1::Vector{Float64}
    e1::Vector{Float64}
    i1::Vector{Float64}
    r1::Vector{Float64}
    out::Vector{Float64}
    M::SparseMatrixCSC{Float64,Int64}
    Mt::SparseMatrixCSC{Float64,Int64}
    natural_rt::Float64
    tinc::Float64
    tinf::Float64
    function SEIR_Parameters(ndays, s1, e1, i1, r1)
        ls1 = length(s1)
        @assert length(e1) == ls1
        @assert length(i1) == ls1
        @assert length(r1) == ls1

        out = ones(ls1)
        M = spzeros(ls1, ls1)
        Mt = spzeros(ls1, ls1)
        new(ndays, ls1, s1, e1, i1, r1, out, M, Mt, 2.5, 5.2, 2.9)
    end
    function SEIR_Parameters(ndays, s1, e1, i1, r1, out, M, Mt)
        ls1 = length(s1)
        @assert length(e1) == ls1
        @assert length(i1) == ls1
        @assert length(r1) == ls1
        @assert size(M) == (ls1, ls1)
        @assert all(M .>= 0.0)
        @assert size(Mt) == (ls1, ls1)
        @assert all(Mt .>= 0.0)
        @assert size(out) == (ls1,)
        @assert all(out .>= 0.0)

        new(ndays, ls1, s1, e1, i1, r1, out, M, Mt, 2.5, 5.2, 2.9)
    end
end


"""
    struct SEIR_Parameters

    Parameters to define a SEIR model:

    ndays: for the simulation duration.
    ncities: number of (interconnected) cities in the model.
    s1, e1, i1, r1: start proportion of the population in each SEIR class.
    out: vector that represents the proportion of the population that leaves
        each city during the day.
    M: Matrix that has at position (i, j) how much population goes from city
        i to city j, the proportion with respect to the population of the
        destiny (j).
    Mt: transpose of M.
    natural_rt: The natural reproduction rate of the disease, right now it
        is assumed constant related to Covid19 (2.5).
    tinc: incubation time related to Covid19 (5.2).
    tinf: time of infection related to Covid19 (2.9).
"""
function seir_model_with_free_initial_values(prm)
    # Save work and get col indices of both M and Mt
    coli_M = [findnz(prm.M[:,c])[1] for c in 1:prm.ncities]
    coli_Mt = [findnz(prm.Mt[:,c])[1] for c in 1:prm.ncities]

    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 5, "linear_solver" => "ma97"))
    # For simplicity I am assuming that one step per day is OK.
    dt = 1.0

    # Note that I do not fix the initial state. It should be defined elsewhere.
    # State variables
    @variable(m, 0.0 <= s[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= e[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= i[1:prm.ncities, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= r[1:prm.ncities, 1:prm.ndays] <= 1.0)

    # Control variable
    @variable(m, 0.0 <= rt[1:prm.ncities, 1:prm.ndays] <= prm.natural_rt)

    # Expressions that define "sub-states"
    # enter denotes the proportion of the population that enter city c during
    # the day.
    @NLexpression(m, enter[c=1:prm.ncities, t=1:prm.ndays],
        sum(prm.M[k, c]*(1.0 - i[k, t]) for k in coli_M[c])
    )
    # p_day is the ratio that the population of a city varies during the day
    @NLexpression(m, p_day[c=1:prm.ncities, t=1:prm.ndays],
        (1.0 - prm.out[c]) + prm.out[c]*i[c, t] + enter[c, t]
    )

    # Parameter that measures how much important is the infection during the day
    # when compared to the night.
    α = 2/3

    # Implement a vectorized version of Heun's method.

    # Compute the gradients at time t of the SEIR model.
    @NLexpression(m, t1[c=1:prm.ncities, t=1:prm.ndays],
        sum(rt[k, t]*prm.Mt[k, c]*s[c, t]*i[k, t]/p_day[k, t] for k = coli_Mt[c])
    )
    @NLexpression(m, ds[c=1:prm.ncities, t=1:prm.ndays],
        -1.0/prm.tinf*(
         α*( rt[c, t]*(1.0 - prm.out[c])*s[c, t]*i[c, t]/p_day[c, t] +
             t1[c, t] ) +
         (1 - α)*rt[c, t]*s[c, t]*i[c, t])
    )
    @NLexpression(m, de[c=1:prm.ncities, t=1:prm.ndays],
        -ds[c, t] - (1.0/prm.tinc)*e[c,t]
    )
    @NLexpression(m, di[c=1:prm.ncities, t=1:prm.ndays],
        (1.0/prm.tinc)*e[c, t] - (1.0/prm.tinf)*i[c, t]
    )
    @NLexpression(m, dr[c=1:prm.ncities, t=1:prm.ndays],
        (1.0/prm.tinf)*i[c, t]
    )

    # Do the Euler step from the point in time t - 1 computing the intermediate
    # point that we express as ?p (p is for plus).
    @NLexpression(m, sp[c=1:prm.ncities, t=2:prm.ndays], s[c, t - 1] + ds[c, t - 1]*dt)
    @NLexpression(m, ep[c=1:prm.ncities, t=2:prm.ndays], e[c, t - 1] + de[c, t - 1]*dt)
    @NLexpression(m, ip[c=1:prm.ncities, t=2:prm.ndays], i[c, t - 1] + di[c, t - 1]*dt)
    @NLexpression(m, rp[c=1:prm.ncities, t=2:prm.ndays], r[c, t - 1] + dr[c, t - 1]*dt)

    # Compute the gradients in the intermediate point.
    @NLexpression(m, t2[c=1:prm.ncities, t=2:prm.ndays],
        sum(rt[k, t]*prm.Mt[k, c]*sp[c, t]*ip[k, t]/p_day[k, t] for k = coli_Mt[c])
    )
    @NLexpression(m, dsp[c=1:prm.ncities, t=2:prm.ndays],
        -1.0/prm.tinf*(
        α*( rt[c, t]*(1.0 - prm.out[c])*sp[c, t]*ip[c, t]/p_day[c, t] +
            t2[c, t] ) +
        (1 - α)*rt[c, t]*sp[c, t]*ip[c, t])
    )
    @NLexpression(m, dep[c=1:prm.ncities, t=2:prm.ndays],
        -dsp[c, t] - (1.0/prm.tinc)*ep[c,t]
    )
    @NLexpression(m, dip[c=1:prm.ncities, t=2:prm.ndays],
        (1.0/prm.tinc)*ep[c, t] - (1.0/prm.tinf)*ip[c, t]
    )
    @NLexpression(m, drp[c=1:prm.ncities, t=2:prm.ndays],
        (1.0/prm.tinf)*ip[c, t]
    )

    # Perform a Heun's update
    @NLconstraint(m, [c=1:prm.ncities, t=2:prm.ndays], s[c, t] == s[c, t - 1] + 0.5*(ds[c, t - 1] + dsp[c, t])*dt)
    @NLconstraint(m, [c=1:prm.ncities, t=2:prm.ndays], e[c, t] == e[c, t - 1] + 0.5*(de[c, t - 1] + dep[c, t])*dt)
    @NLconstraint(m, [c=1:prm.ncities, t=2:prm.ndays], i[c, t] == i[c, t - 1] + 0.5*(di[c, t - 1] + dip[c, t])*dt)
    @NLconstraint(m, [c=1:prm.ncities, t=2:prm.ndays], r[c, t] == r[c, t - 1] + 0.5*(dr[c, t - 1] + drp[c, t])*dt)

    return m
end

"""
    seir_model(prm)

    Creates a SEIR model setting the initial parameters for the SEIR variables.
"""
function seir_model(prm)
    m = seir_model_with_free_initial_values(prm)

    # Initial state
    s1, e1, i1, r1 = m[:s][:, 1], m[:e][:, 1], m[:i][:, 1], m[:r][:, 1]
    for c in 1:prm.ncities
        fix(s1[c], prm.s1[c]; force=true)
        fix(e1[c], prm.e1[c]; force=true)
        fix(i1[c], prm.i1[c]; force=true)
        fix(r1[c], prm.r1[c]; force=true)
    end
    return m
end


"""
    fixed_rt_model

    Creates a model fixing all initial parameters and defining the RT
    as the natural R0 for Covid19.
"""
function fixed_rt_model(prm)
    m = seir_model(prm)
    rt = m[:rt]

    # Fix all rts
    for c = 1:prm.ncities
        for t = 1:prm.ndays
            fix(rt[c, t], prm.natural_rt; force=true)
        end
    end
    return m
end

"""
    fit_initcond_model(prm, initial_data)

    Fit the initial parameters for a single city.
"""
function fit_initial(data)

    prm = SEIR_Parameters(length(data), [0.0], [0.0], [0.0], [0.0])

    m = seir_model_with_free_initial_values(prm)

    # Initial state
    s1, e1, i, r1, rt = m[:s][1], m[:e][1], m[:i], m[:r][1], m[:rt]
    fix(r1, prm.r1; force=true)
    set_start_value(s1, prm.s1)
    set_start_value(e1, prm.e1)
    set_start_value(i[1], prm.i1)
    for t = 1:prm.ndays
        fix(rt[t], prm.natural_rt; force=true)
    end

    @constraint(m, s1 + e1 + i[1] + r1 == 1.0)
    # Compute a scaling factor so as a least square objective makes more sense
    factor = 1.0/mean(initial_data)
    @objective(m, Min, sum((factor*(i[t] - data[t]))^2 for t = 1:prm.ndays))

    return m
end

"""
    control_multcities

    Built a simple control problem that tries to force the proportion
    of infected to remain below target every day for every city.
"""
function control_multcities(s1, e1, r1, i1, out, M, population, ndays, target, min_rt=1.0)
    prm = SEIR_Parameters(ndays, s1, e1, i1, r1, out, sparse(M), sparse(M'))

    m = seir_model(prm)

    rt = m[:rt]
    for c=1:prm.ncities, d=1:prm.ndays
        set_lower_bound(rt[c, d], min_rt)
    end

    # Compute the total variation of RT
    rt = m[:rt]
    @variable(m, tot_var[c=1:prm.ncities, t=2:prm.ndays])
    @constraint(m, tot_var1[c=1:prm.ncities, t=2:prm.ndays], rt[c, t - 1] - rt[c, t] <= tot_var[c, t])
    @constraint(m, tot_var2[c=1:prm.ncities, t=2:prm.ndays], rt[c, t] - rt[c, t - 1] <= tot_var[c, t])

    # Constraint on maximum level of infection
    i = m[:i]
    @constraint(m, [c=1:prm.ncities, t=2:prm.ndays], i[c, t] <= target[c])

    mean_population = mean(sqrt.(population))
    @objective(m, Min,
        # Try to keep as many people working as possible
        sum(sqrt(population[c])/mean_population*(prm.natural_rt - rt[c, d]) for c = 1:prm.ncities for d = 1:prm.ndays) +
        # Avoid large variations
        sum(tot_var) -
        # Try to enforce different cities to alternate the controls
        0.1/(prm.ncities)*sum((sqrt(population[c]) + sqrt(population[cl]))/(mean_population*d^0.25)*(rt[c, d] - rt[cl, d])^2 
            for c = 1:prm.ncities for cl = c + 1:prm.ncities for d = 1:prm.ndays)
    )

    return m
end

"""
    window_control_multcities

    Try to attain a maximum number of infected in all cities below
    target but only allow changes of the control in the beggining of
    a time window.
"""
function window_control_multcities(s1, e1, r1, i1, out, M, population, ndays, target, window, min_rt=1.0)
    prm = SEIR_Parameters(ndays, s1, e1, i1, r1, out, sparse(M), sparse(M'))
    m = seir_model(prm)

    # Add constraints to make the dynamics interesting
    rt = m[:rt]
    i = m[:i]

    # Allow piece wise constants controls
    for c=1:prm.ncities, d=1:prm.ndays
        set_lower_bound(rt[c, d], min_rt)
    end

    for d = 1:window:prm.ndays
        @constraint(m, [c=1:prm.ncities, dl=d + 1:minimum([d + window - 1, prm.ndays])],
            rt[c, dl] == rt[c, d]
        )
    end

    # Bound the maximal infection rate
    @constraint(m, [c=1:prm.ncities, t=2:prm.ndays], i[c, t] <= target[c])

    # Total difference between decisions
    @variable(m, dif_rt[c=1:prm.ncities, cl=c + 1:prm.ncities, d=1:window:prm.ndays])
    @constraint(m, [c=1:prm.ndays, cl=c+1:prm.ncities, d=1:window:prm.ndays], rt[c, d] - rt[cl, d] <= dif_rt[c, cl, d])
    @constraint(m, [c=1:prm.ndays, cl=c+1:prm.ncities, d=1:window:prm.ndays], rt[cl, d] - rt[c, d] <= dif_rt[c, cl, d])

    mean_population = mean(sqrt.(population))
    @objective(m, Min,
        # Try to keep as many people working as possible
        sum(sqrt(population[c])/mean_population*(prm.natural_rt - rt[c, d]) 
            for c = 1:prm.ncities for d = 1:prm.ndays) -
        # Try to enforce different cities to alternate the controls
        window/(prm.ncities)*sum((sqrt(population[c]) + sqrt(population[cl]))/(mean_population*d^0.25)*(rt[c, d] - rt[cl, d])^2 
            for c = 1:prm.ncities for cl = c + 1:prm.ncities for d = 1:window:prm.ndays)
    )

    return m
end


"""
    playground

    Placeholder for doing different tests.
"""
function playground(s1, e1, i1, r1, out, M, population, ndays, control, target)
    prm = SEIR_Parameters(ndays, s1, e1, i1, r1, out, sparse(M), sparse(M'))
    m = seir_model(prm)

    rt = m[:rt]
    for c=1:prm.ncities, d=1:prm.ndays
        fix(rt[c, d], control[c, d]; force=true)
    end

    # Compute the total variation of RT
    rt = m[:rt]
    @variable(m, tot_var[c=1:prm.ncities, t=2:prm.ndays])
    @constraint(m, tot_var1[c=1:prm.ncities, t=2:prm.ndays], rt[c, t - 1] - rt[c, t] <= tot_var[c, t])
    @constraint(m, tot_var2[c=1:prm.ncities, t=2:prm.ndays], rt[c, t] - rt[c, t - 1] <= tot_var[c, t])

    # Constraint on maximum level of infection
    i = m[:i]
    @constraint(m, [c=1:prm.ncities, t=2:prm.ndays], i[c, t] <= target)

    mean_population = mean(sqrt.(population))
    @objective(m, Min,
        # Try to keep as many people working as possible
        sum(sqrt(population[c])/mean_population*(prm.natural_rt - rt[c, d]) for c = 1:prm.ncities for d = 1:prm.ndays) +
        # Avoid large variations
        sum(tot_var) -
        # Try to enforce different cities to alternate the controls
        0.1/(prm.ncities)*sum((sqrt(population[c]) + sqrt(population[cl]))/(mean_population*d^0.25)*(rt[c, d] - rt[cl, d])^2 
            for c = 1:prm.ncities for cl = c + 1:prm.ncities for d = 1:prm.ndays)
    )

    return m
end
