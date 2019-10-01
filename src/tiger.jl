abstract type AbstractTigerPOMDP{O} <: POMDP{Symbol, Symbol, O} end

@with_kw struct COTigerPOMDP <: AbstractTigerPOMDP{Float64}
    r_listen1::Float64         = -1.0
    r_listen2::Float64         = -2.0
    r_findtiger::Float64        = -100.0
    r_escapetiger::Float64      = 10.0
    p_correct_1::Float64        = 0.6
    p_correct_2::Float64        = 0.9
    discount::Float64           = 0.95
end

@with_kw struct DOTigerPOMDP <: AbstractTigerPOMDP{Symbol}
    r_listen1::Float64         = -1.0
    r_listen2::Float64         = -2.0
    r_findtiger::Float64        = -100.0
    r_escapetiger::Float64      = 10.0
    p_correct_1::Float64        = 0.6
    p_correct_2::Float64        = 0.9
    discount::Float64           = 0.95
end

const stateinds = (left=1, right=2, done=3)
const actioninds = (left=1, right=2, listen1=3, listen2=4)
const obsinds = (left=1, right=2)

POMDPs.states(::AbstractTigerPOMDP) = keys(stateinds)
POMDPs.actions(::AbstractTigerPOMDP) = keys(actioninds)
POMDPs.observations(::DOTigerPOMDP) = keys(obsinds)

POMDPs.n_states(::AbstractTigerPOMDP) = length(stateinds)
POMDPs.n_actions(::AbstractTigerPOMDP) = length(actioninds)
POMDPs.n_observations(::DOTigerPOMDP) = length(obsinds)

POMDPs.stateindex(::AbstractTigerPOMDP, s) = stateinds[s]
POMDPs.actionindex(::AbstractTigerPOMDP, a) = actioninds[a]
POMDPs.obsindex(::DOTigerPOMDP, o) = obsinds[o]

POMDPs.initialstate_distribution(m::AbstractTigerPOMDP) = POMDPModelTools.Uniform((:left, :right))

function POMDPs.transition(m::AbstractTigerPOMDP, s, a)
    if a in (:left, :right)
        return Deterministic(:done)
    else
        return Deterministic(s)
    end
end

POMDPs.isterminal(m::AbstractTigerPOMDP, s) = s == :done

function POMDPs.observation(m::DOTigerPOMDP, a, sp)
    if a == :listen1
        probs = (m.p_correct_1, 1.0-m.p_correct_1)
    elseif a == :listen2
        probs = (m.p_correct_2, 1.0-m.p_correct_2)
    else # a is a door open
        probs = (0.5, 0.5)
    end
    if sp == :left
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
    # r = rand(rng)
    # sum = 0.0
    # ind::Int
    # for i in 1:length(d.probs)
    #     sum += d.probs[i]
    #     if r <= sum
    #         ind = i
    #         break
    #     end
    # end
    # return rand(rng, d.dists[i])
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
function POMDPs.observation(m::COTigerPOMDP, a::Symbol, sp::Symbol)
    if a == :listen1
        probs = (m.p_correct_1, 1.0-m.p_correct_1)
    elseif a == :listen2
        probs = (m.p_correct_2, 1.0-m.p_correct_2)
    else # a is a door open
        probs = (0.5, 0.5)
    end
    if sp == :left
        dists = (Distributions.Uniform(0.0, 0.5), Distributions.Uniform(0.5, 1.0))
    else
        dists = (Distributions.Uniform(0.5, 1.0), Distributions.Uniform(0.0, 0.5))
    end
    return MultiUniformDistribution(probs, dists)
end

function POMDPs.reward(m::AbstractTigerPOMDP, s, a)
    if s == :done
        return 0.0
    end
    if a == :listen1
        return m.r_listen1
    elseif a == :listen2
        return m.r_listen2
    elseif s == a # a is open
        return m.r_findtiger
    else
        return m.r_escapetiger
    end
end

POMDPs.discount(m::AbstractTigerPOMDP) = m.discount
