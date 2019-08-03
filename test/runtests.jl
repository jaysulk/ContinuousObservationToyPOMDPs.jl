using Test
using ContinuousObservationToyPOMDPs
using ParticleFilters
using QMDP
using POMDPs
using POMDPSimulators

sld = SimpleLightDark()
@test isterminal(sld, sld.radius+1)
p = solve(LDHSolver(), sld)
filter = SIRParticleFilter(sld, 1000)
for (s, b, a, r, sp, o) in stepthrough(sld, p, filter, "sbarspo", max_steps=10)
    @show (s, a, r, sp, o)
    @show mean(b)
end

qp = solve(QMDPSolver(), sld, verbose=true)
for (s, b, a, r, sp, o) in stepthrough(sld, qp, "sbarspo", max_steps=10)
    @show (s, a, r, sp, o)
    @show b
end

m = COTigerPOMDP()
qp = solve(QMDPSolver(), m)
for (s, b, a, r, sp, o, t) in stepthrough(m, qp, "sbarspot", max_steps=100)
    @show (t=t, s=s, a=a, r=r, sp=sp, o=o)
    @show collect(s=>pdf(b, s) for s in support(b))
end
