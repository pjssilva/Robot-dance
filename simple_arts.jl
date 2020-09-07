"""

Simple auto-regressive time series.

"""

using LinearAlgebra
using Distributions


mutable struct Simple_ARTS
    rhomin::Float64
    rhomax::Float64
    Δ::Float64
    c0::Vector{Float64}
    c1::Vector{Float64}
    A::Matrix{Float64}
    σω::Float64
    s0::Vector{Float64}
    t::Int64
    state::Vector{Float64}
    Ak::Matrix{Float64}
    sumΘ2::Float64
    F1p::Float64

    function Simple_ARTS(rhomin, rhomax, c0, c1, ϕ, σω, s0, p)
        Δ = rhomax - rhomin
        n = length(ϕ)
        vec_c0 = zeros(n)
        vec_c0[1] = c0
        vec_c1 = zeros(n)
        vec_c1[1] = c1
        A = zeros(n, n)
        A[1, :] .= ϕ
        A[2:end, 1] .= 1.0
        Ak = Matrix{Float64}(I, n, n)
        state = (s0 .- rhomin) ./ Δ
        sumΘ2 = 0.0
        F1p = quantile(Normal(), 1.0 - p)

        new(rhomin, rhomax, Δ, vec_c0, vec_c1, A, σω, s0, 0, state, Ak, sumΘ2,  F1p)
    end
end

function reset(arts::Simple_ARTS)
    arts.t = 0
    n = length(arts.state)
    arts.Ak = Matrix{Float64}(I, n, n)
    arts.state = (arts.s0 .- arts.rhomin) ./ arts.Δ
    arts.sumΘ2 = 0.0
    nothing
end

function iterate(arts::Simple_ARTS)
    arts.t += 1
    arts.sumΘ2 += arts.Ak[1, 1]*arts.Ak[1, 1]
    arts.Ak *= arts.A
    arts.state = arts.c0 + arts.c1*arts.t + arts.A*arts.state
    return arts.rhomin + arts.Δ*arts.state[1], arts.F1p*arts.σω*arts.Δ*sqrt(arts.sumΘ2)
end
        

