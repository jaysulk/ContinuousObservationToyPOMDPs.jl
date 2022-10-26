module ContinuousObservationToyPOMDPs

using POMDPs
using POMDPModelTools
using Parameters
using Random
using POMDPPolicies
using QMDP
using ParticleFilters
using StaticArrays
using Statistics
using Distributions
using Base.Iterators

export
    COTigerPOMDP,
    DOTigerPOMDP,
    TimedCOTigerPOMDP,
    TimedDOTigerPOMDP

include("tiger.jl")

end # module
