@with_kw struct COTigerPOMDP <: POMDP{Symbol, Symbol, Float64}
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

POMDPs.states(::COTigerPOMDP) = keys(stateinds)
POMDPs.actions(::COTigerPOMDP) = keys(actioninds)

POMDPs.n_states(::COTigerPOMDP) = length(stateinds)
POMDPs.n_actions(::COTigerPOMDP) = length(actioninds)

POMDPs.stateindex(::COTigerPOMDP, s) = stateinds[s]
POMDPs.actionindex(::COTigerPOMDP, a) = actioninds[a]

POMDPs.initialstate_distribution(m::COTigerPOMDP) = POMDPModelTools.Uniform((:left, :right))

function POMDPs.transition(m::COTigerPOMDP, s, a)
    if a in (:left, :right)
        return Deterministic(:done)
    else
        return Deterministic(s)
    end
end

POMDPs.isterminal(m::COTigerPOMDP, s) = s == :done

struct MultiUniformDistribution{N}
    probs::NTuple{N, Float64}
    dists::NTuple{N, Uniform}
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
        dists = (Uniform(0, 0.5), Uniform(0.5, 1))
    else
        dists = (Uniform(0.5, 1), Uniform(0, 0.5))
    end
    return MultiUniformDistribution(probs, dists)
end

function POMDPs.reward(m::COTigerPOMDP, s, a)
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

POMDPs.discount(m::COTigerPOMDP) = m.discount
