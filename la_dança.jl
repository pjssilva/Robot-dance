using JuMP
using Ipopt
using Printf
using Plots

struct SEIR_Parameters
    ndays::Int64
    s0::Float64
    e0::Float64
    i0::Float64
    r0::Float64
    natural_r::Float64
    tinc::Float64
    tinf::Float64
    SEIR_Parameters(ndays, s0, e0, i0, r0) = new(ndays, s0, e0, i0, r0, 2.5, 5.2, 2.9)
end

function seir_grad(m, s, e, i, r, rt, tinf, tinc)
    ds = @NLexpression(m, -(rt/tinf)*s*i)
    de = @NLexpression(m, (rt/tinf)*s*i - (1.0/tinc)*e)
    di = @NLexpression(m, (1.0/tinc)*e - (1.0/tinf)*i)
    dr = @NLexpression(m, (1.0/tinf)*i)
    return ds, de, di, dr
end


function seir_model_with_free_initial_values(iparam)
    m = Model(Ipopt.Optimizer)
    # For simplicity I am assuming that one step per day is OK.
    dt = 1.0

    # State variables
    lastday = iparam.ndays - 1
    @variable(m, 0.0 <= s[0:lastday] <= 1.0)
    @variable(m, 0.0 <= e[0:lastday] <= 1.0)
    @variable(m, 0.0 <= i[0:lastday] <= 1.0)
    @variable(m, 0.0 <= r[0:lastday] <= 1.0)
    @variable(m, 0.0 <= rt[0:lastday] <= iparam.natural_r)

    # Note that I do not fix the initial state. It should be defined elsewhere.

    # Implement Haun's method
    for t = 1:lastday
        ds, de, di, dr = seir_grad(m, s[t - 1], e[t - 1], i[t - 1], r[t - 1], rt[t - 1], iparam.tinf, iparam.tinc)

        sp = @NLexpression(m, s[t - 1] + ds*dt)
        ep = @NLexpression(m, e[t - 1] + de*dt)
        ip = @NLexpression(m, i[t - 1] + di*dt)
        rp = @NLexpression(m, r[t - 1] + dr*dt)

        dsp, dep, dip, drp = seir_grad(m, sp, ep, ip, rp, rt[t], iparam.tinf, iparam.tinc)

        @NLconstraint(m, s[t] == s[t - 1] + 0.5*(ds + dsp)*dt)
        @NLconstraint(m, e[t] == e[t - 1] + 0.5*(de + dep)*dt)
        @NLconstraint(m, i[t] == i[t - 1] + 0.5*(di + dip)*dt)
        @NLconstraint(m, r[t] == r[t - 1] + 0.5*(dr + drp)*dt)
    end
    return m
end

function seir_model(iparam)
    m = seir_model_with_free_initial_values(iparam)

    # Initial state
    s0, e0, i0, r0 = m[:s][0], m[:e][0], m[:i][0], m[:r][0]
    fix(s0, iparam.s0; force=true)
    fix(e0, iparam.e0; force=true)
    fix(i0, iparam.i0; force=true)
    fix(r0, iparam.r0; force=true)

    return m
end


function control_rt_model(iparam, max_i)
    lastday = iparam.ndays - 1
    m = seir_model(iparam)
    rt, i = m[:rt], m[:i]

    # Rts can not change too fast.
    for t = 1:lastday
        @constraint(m, 0.95*rt[t - 1] <= rt[t])
        @constraint(m, rt[t] <= 1.05*rt[t - 1])
    end

    # Limit infection
    for t = 0:lastday
        @constraint(m, i[t] <= max_i)
    end

    # Maximize rt
    @objective(m, Max, sum(rt))

    return m
end


function fixed_rt_model(iparam)
    lastday = iparam.ndays - 1
    m = seir_model(iparam)
    rt = m[:rt]

    # Fix all rts
    for t = 1:lastday
        fix(rt[t], iparam.natural_r; force=true)
    end
    return m
end


function fit_initcond_model(iparam, initial_data)
    m = seir_model_with_free_initial_values(iparam)
    lastday = iparam.ndays - 1

    # Initial state
    s0, e0, i, r0, rt = m[:s][0], m[:e][0], m[:i], m[:r][0], m[:rt]
    fix(r0, iparam.r0; force=true)
    set_start_value(s0, iparam.s0)
    set_start_value(e0, iparam.e0)
    set_start_value(i[0], iparam.i0)
    for t = 0:lastday
        fix(rt[t], iparam.natural_r; force=true)
    end

    @constraint(m, s0 + e0 + i[0] + r0 == 1.0)
    @objective(m, Min, sum((i[t] - initial_data[t + 1])^2 for t = 0:lastday))

    return m
end


function test_control_rt(iparam)
    #iparam = SEIR_Parameters(365, 0.9999999183808258, 1.632383484751865e-06, 8.161917423759325e-07, 0.0)

    # First the uncontroled infection
    m = fixed_rt_model(iparam)
    optimize!(m)
    iv = value.(m[:i]).data
    plot(iv, label="Uncontroled", lw=2)
    max_inf = maximum(iv)

     # Now controled infections
     targets = [0.01, 0.03, 0.05]
     for max_infection in targets
         m = control_rt_model(iparam, max_infection)
         optimize!(m)
         plot!(value.(m[:i]).data, label="$max_infection", lw=2)
    end

    title!("Controlled Rt")
    xlabel!("days")
    ylabel!("rt")
    ylims!(0.0, max_inf + 0.005)
    yticks!(vcat([0.0], targets, [max_inf]), yformatter = yi -> @sprintf("%.2f", yi))
    xticks!([0, 100, 200, 300, 365])
end
