abstract type AbstractTigerPOMDP{S, O} <: POMDP{S, Symbol, O} end

# Infinite Horizon

@with_kw struct COTigerPOMDP <: AbstractTigerPOMDP{Symbol, Float64}
    r_wait::Float64             = -1.0
    r_listen::Float64           = -2.0
    r_findtiger::Float64        = -100.0
    r_escapetiger::Float64      = 10.0
    p_correct::Float64          = 0.85
    discount::Float64           = 0.95
end

@with_kw struct DOTigerPOMDP <: AbstractTigerPOMDP{Symbol, Symbol}
    r_wait::Float64             = -1.0
    r_listen::Float64           = -2.0
    r_findtiger::Float64        = -100.0
    r_escapetiger::Float64      = 10.0
    p_correct::Float64          = 0.85
    discount::Float64           = 0.95
end

# Finite Horizon

abstract type AbstractTimedTigerPOMDP{O} <: AbstractTigerPOMDP{Tuple{Symbol,Int}, O} end

@with_kw struct TimedCOTigerPOMDP <: AbstractTimedTigerPOMDP{Float64}
    r_wait::Float64             = -1.0
    r_listen::Float64           = -2.0
    r_findtiger::Float64        = -100.0
    r_escapetiger::Float64      = 10.0
    p_correct::Float64          = 0.85
    horizon::Int                = 5
    discount::Float64           = 0.95
end

@with_kw struct TimedDOTigerPOMDP <: AbstractTimedTigerPOMDP{Symbol}
    r_wait::Float64             = -1.0
    r_listen::Float64           = -2.0
    r_findtiger::Float64        = -100.0
    r_escapetiger::Float64      = 10.0
    p_correct::Float64          = 0.85
    horizon::Int                = 5
    discount::Float64           = 0.95
end

const ContinuousTiger = Union{COTigerPOMDP, TimedCOTigerPOMDP}
const DiscreteTiger = Union{DOTigerPOMDP, TimedDOTigerPOMDP}

const stateinds = (left=1, right=2, done=3)
const tigerinds = (left=1, right=2)
const actioninds = (left=1, right=2, wait=3, listen=4)
const obsinds = (left=1, right=2)
tigerpos(s::Symbol) = s
tigerpos(st::Tuple{Symbol,Int}) = st[1]
stepindex(st::Tuple{Symbol,Int}) = st[2]

horizon(m::AbstractTimedTigerPOMDP) = m.horizon

POMDPs.states(::AbstractTigerPOMDP) = keys(stateinds)
POMDPs.actions(::AbstractTigerPOMDP) = keys(actioninds)
POMDPs.states(m::AbstractTimedTigerPOMDP) = product(keys(tigerinds), 0:horizon(m))
POMDPs.observations(::DOTigerPOMDP) = keys(obsinds)

POMDPs.stateindex(::AbstractTigerPOMDP, s) = stateinds[s]
function POMDPs.stateindex(m::AbstractTimedTigerPOMDP, s)
    LinearIndices((length(tigerinds), horizon(m)+1))[tigerinds[tigerpos(s)], stepindex(s)+1]
end
POMDPs.actionindex(::AbstractTigerPOMDP, a) = actioninds[a]
POMDPs.obsindex(::DOTigerPOMDP, o) = obsinds[o]

POMDPs.initialstate_distribution(m::AbstractTigerPOMDP) = POMDPModelTools.Uniform((:left, :right))
function POMDPs.initialstate_distribution(m::AbstractTimedTigerPOMDP)
    POMDPModelTools.Uniform((t, 0) for t in keys(tigerinds))
end

function POMDPs.transition(m::AbstractTigerPOMDP, s, a)
    if a in (:left, :right)
        return Deterministic(:done)
    else
        return Deterministic(s)
    end
end

POMDPs.transition(m::AbstractTimedTigerPOMDP, s, a) = Deterministic((tigerpos(s), stepindex(s)+1))

POMDPs.isterminal(m::AbstractTigerPOMDP, s) = s == :done
POMDPs.isterminal(m::AbstractTimedTigerPOMDP, s) = stepindex(s) >= horizon(m)

function POMDPs.observation(m::DiscreteTiger, a, sp)
    if a == :listen
        probs = (m.p_correct, 1.0-m.p_correct)
    else # a is a door open or wait
        probs = (0.5, 0.5)
    end
    if tigerpos(sp) == :left
        obs = (:left, :right)
    else
        obs = (:right, :left)
    end
    return SparseCat(obs, probs)
end

struct MultiUniformDistribution{N}
    probs::NTuple{N, Float64}
    dists::NTuple{N, Distributions.Uniform{Float64}}
end

function POMDPs.rand(rng::AbstractRNG, d::MultiUniformDistribution)
    i = rand(rng, Categorical(SVector(d.probs)))
    return rand(rng, d.dists[i])
end

function POMDPs.pdf(d::MultiUniformDistribution, o)
    for i in 1:length(d.probs)
        u = d.dists[i]
        if minimum(u) <= o <= maximum(u)
            return pdf(u, o)*d.probs[i]
        end
    end
    return 0.0
end

# left = Uniform(0, 0.5), right = Uniform(0.5, 1)
function POMDPs.observation(m::ContinuousTiger, a, sp)
    if a == :listen
        probs = (m.p_correct, 1.0-m.p_correct)
    else # a is a door open or wait
        probs = (0.5, 0.5)
    end
    if tigerpos(sp) == :left
        dists = (Distributions.Uniform(0.0, 0.5), Distributions.Uniform(0.5, 1.0))
    else
        dists = (Distributions.Uniform(0.5, 1.0), Distributions.Uniform(0.0, 0.5))
    end
    return MultiUniformDistribution(probs, dists)
end

function POMDPs.reward(m::AbstractTigerPOMDP, s, a)
    if isterminal(m, s)
        return 0.0
    end
    if a == :wait
        return m.r_wait
    elseif a == :listen
        return m.r_listen
    elseif s == a # a is open
        return m.r_findtiger
    else
        return m.r_escapetiger
    end
end

POMDPs.discount(m::AbstractTigerPOMDP) = m.discount
