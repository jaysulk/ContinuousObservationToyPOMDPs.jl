@with_kw struct SimpleLightDark <: POMDPs.POMDP{Int,Int,Float64}
    discount::Float64       = 0.95
    correct_r::Float64      = 100.0
    incorrect_r::Float64    = -100.0
    light_loc::Int          = 10
    radius::Int             = 60
end
POMDPs.discount(p::SimpleLightDark) = p.discount
POMDPs.isterminal(p::SimpleLightDark, s::Number) = !(s in -p.radius:p.radius)

const ACTIONS = [-10, -1, 0, 1, 10]
POMDPs.actions(p::SimpleLightDark) = ACTIONS
POMDPs.n_actions(p::SimpleLightDark) = length(actions(p))
const ACTION_INDS = Dict(a=>i for (i,a) in enumerate(actions(SimpleLightDark())))
POMDPs.actionindex(p::SimpleLightDark, a::Int) = ACTION_INDS[a]

POMDPs.states(p::SimpleLightDark) = -p.radius:p.radius + 1
POMDPs.n_states(p::SimpleLightDark) = length(states(p))
POMDPs.stateindex(p::SimpleLightDark, s::Int) = s+p.radius+1

function POMDPs.transition(p::SimpleLightDark, s::Int, a::Int) 
    if a == 0
        return SparseCat(SVector(p.radius+1), SVector(1.0))
    else
        return SparseCat(SVector(clamp(s+a, -p.radius, p.radius)), SVector(1.0))
    end
end

POMDPs.observation(p::SimpleLightDark, sp) = Normal(sp, abs(sp - p.light_loc) + 0.0001)

function POMDPs.reward(p::SimpleLightDark, s, a)
    if a == 0
        return s == 0 ? p.correct_r : p.incorrect_r
    else
        return -1.0
    end
end

function POMDPs.initialstate_distribution(p::SimpleLightDark)
    ps = ones(2*div(p.radius,2)+1)
    ps /= length(ps)
    return SparseCat(div(-p.radius,2):div(p.radius,2), ps)
end

@with_kw struct DSimpleLightDark <: POMDPs.POMDP{Int, Int, Int}
    sld::SimpleLightDark = SimpleLightDark()
    binsize::Float64     = 1.0
end

POMDPs.generate_o(p::DSimpleLightDark, sp, rng::AbstractRNG) = floor(Int, rand(rng, observation(p.sld, sp))/p.binsize)
POMDPs.generate_o(p::DSimpleLightDark, a, sp, rng::AbstractRNG) = generate_o(p, sp, rng)
function POMDPs.generate_sor(p::DSimpleLightDark, s, a, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    o = generate_o(p, sp, rng)
    r = reward(p, s, a, sp)
    return sp, o, r
end

function POMDPModelTools.obs_weight(p::DSimpleLightDark, sp::Int, o::Int)
    cod = observation(p.sld, sp)
    return cdf(cod, (o+1)*p.binsize) - cdf(cod, o*p.binsize)
end

POMDPs.discount(p::DSimpleLightDark) = discount(p.sld)
POMDPs.isterminal(p::DSimpleLightDark, s::Number) = isterminal(p.sld, s)
POMDPs.actions(p::DSimpleLightDark) = actions(p.sld)
POMDPs.n_actions(p::DSimpleLightDark) = n_actions(p.sld)
POMDPs.action_index(p::DSimpleLightDark, a::Int) = action_index(p.sld, a)
POMDPs.states(p::DSimpleLightDark) = states(p.sld)
POMDPs.n_states(p::DSimpleLightDark) = n_states(p.sld)
POMDPs.stateindex(p::DSimpleLightDark, s::Int) = state_index(p.sld, s)
POMDPs.transition(p::DSimpleLightDark, s::Int, a::Int) = transition(p.sld, s, a)
POMDPs.reward(p::DSimpleLightDark, s, a) = reward(p.sld, s, a)
POMDPs.initialstate_distribution(p::DSimpleLightDark) = initial_state_distribution(p.sld)

struct LDHeuristic <: Policy
    p::SimpleLightDark
    q::AlphaVectorPolicy{SimpleLightDark, Int}
    std_thresh::Float64
end

struct LDHSolver <: Solver
    q::QMDPSolver
    std_thresh::Float64
end

LDHSolver(;std_thresh::Float64=0.1, kwargs...) = LDHSolver(QMDPSolver(;kwargs...), std_thresh)

POMDPs.solve(sol::LDHSolver, pomdp::SimpleLightDark) = LDHeuristic(pomdp, solve(sol.q, pomdp), sol.std_thresh)

POMDPs.action(p::LDHeuristic, s::Int) = action(p.q, s)
Random.seed!(p::LDHeuristic, s) = p

function POMDPs.action(p::LDHeuristic, b::AbstractParticleBelief)
    s = std(particles(b))
    if s <= p.std_thresh
        return action(p.q, b)
    else
        m = mean(particles(b))
        ll = p.p.light_loc
        if m == ll
            return -1*Int(sign(ll))
        elseif abs(m-ll) >= 10 
            return -10*Int(sign(m-ll))
        else
            return -Int(sign(m-ll))
        end
    end
end

struct LDSide <: Solver end

mutable struct LDSidePolicy{LD} <: Policy
    q::AlphaVectorPolicy{LD, Int}
end

POMDPs.solve(solver::LDSide, pomdp::Union{SimpleLightDark,DSimpleLightDark}) = LDSidePolicy(solve(QMDPSolver(), pomdp))
Random.seed!(p::LDSidePolicy, s) = p

function POMDPs.action(p::LDSidePolicy, b)
    if pdf(b, mode(b)) > 0.9
        return action(p.q, b)
    else
        return 10
    end
end
