export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
import Base.push!

using ReinforcementLearningCore: AbstractLearner, TabularApproximator
using Flux

"""
    TDLearner(;approximator, method, γ=1.0, α=0.01, n=0)

Use temporal-difference method to estimate state value or state-action value.

# Fields
- `approximator` is `<:TabularApproximator`.
- `γ=1.0`, discount rate.
- `method`: only `:SARS` (Q-learning) is supported for the time being.
- `n=0`: the number of time steps used minus 1.
"""
Base.@kwdef mutable struct TDLearner{M,A} <: AbstractLearner where {A<:TabularApproximator,M<:Symbol}
    approximator::A
    γ::Float64 = 1.0 # discount factor
    α::Float64 = 0.01 # learning rate
    n::Int = 0

    function TDLearner(approximator::A, method::Symbol; γ=1.0, α=0.01, n=0) where {A<:TabularApproximator}
        if method ∉ [:SARS, :SARSA]
            @error "Method $method is not supported"
        else
            new{method, A}(approximator, γ, α, n)
        end
    end
end

RLCore.forward(L::TDLearner, s::Int) = RLCore.forward(L.approximator, s)
RLCore.forward(L::TDLearner, s::Int, a::Int) = RLCore.forward(L.approximator, s, a)

Q(app::TabularApproximator, s::Int, a::Int) = RLCore.forward(app, s, a)
Q(app::TabularApproximator, s::Int) = RLCore.forward(app, s)

"""
    bellman_update!(app::TabularApproximator, s::Int, s_plus_one::Int, a::Int, α::Float64, π_::Float64, γ::Float64)

Update the Q-value of the given state-action pair.
"""
function bellman_update!(
    approx::TabularApproximator,
    state::I1,
    next_state::I2,
    action::I3,
    reward::F1,
    γ::Float64, # discount factor
    α::Float64, # learning rate
) where {I1<:Integer,I2<:Integer,I3<:Integer,F1<:AbstractFloat}
    # Q-learning formula following https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/25c4d3888e178c51ed1ff448f36b0fcaf7c1d8e8/src/q_learn.jl#LL63C26-L63C95
    # Terminology following https://en.wikipedia.org/wiki/Q-learning
    estimate_optimal_future_value = maximum(Q(approx, next_state))
    current_value = Q(approx, state, action)
    raw_q_value = (reward + γ * estimate_optimal_future_value - current_value) # Discount factor γ is applied here
    approx.model[action, state] += α * raw_q_value
    return Q(approx, state, action)
end

function bellman_update!(
    approx::TabularApproximator,
    state::I1,
    next_state::I2,
    action::I3,
    next_action::I3,
    reward::F1,
    γ::Float64, # discount factor
    α::Float64, # learning rate
) where {I1<:Integer,I2<:Integer,I3<:Integer,F1<:AbstractFloat}
    next_value = Q(approx, next_state, next_action)
    current_value = Q(approx, state, action)
    raw_q_value = (reward + γ * next_value - current_value) # Discount factor γ is applied here
    approx.model[action, state] += α * raw_q_value
    return Q(approx, state, action)
end

function _optimise!(
    n::I1,
    γ::F, # discount factor
    α::F, # learning rate
    approx::TabularApproximator{Ar},
    state::I2,
    next_state::I2,
    action::I3,
    reward::F,
) where {I1<:Integer,I2<:Integer,I3<:Integer,Ar<:AbstractArray,F<:AbstractFloat}
    bellman_update!(approx, state, next_state, action, reward, γ, α)
end

function _optimise!(
    n::I1,
    γ::F, # discount factor
    α::F, # learning rate
    approx::TabularApproximator{Ar},
    state::I2,
    next_state::I2,
    action::I3,
    next_action::I3,
    reward::F,
) where {I1<:Integer,I2<:Integer,I3<:Integer,Ar<:AbstractArray,F<:AbstractFloat}
    bellman_update!(approx, state, next_state, action, next_action, reward, γ, α)
end

function RLBase.optimise!(
    L::TDLearner{:SARS},
    t::@NamedTuple{state::I1, next_state::I1, action::I2, reward::F2, terminal::Bool},
) where {I1<:Number,I2<:Number,F2<:AbstractFloat}
    _optimise!(L.n, L.γ, L.α, L.approximator, t.state, t.next_state, t.action, t.reward)
end

function RLBase.optimise!(
    L::TDLearner{:SARSA},
    t::@NamedTuple{state::I1, next_state::I1, action::I2, next_action::I2, reward::F2, terminal::Bool},
) where {I1<:Number,I2<:Number,F2<:AbstractFloat}
    _optimise!(L.n, L.γ, L.α, L.approximator, t.state, t.next_state, t.action, t.next_action, t.reward)
end

function RLBase.optimise!(learner::TDLearner, stage::PostActStage, trajectory::Trajectory)
    idx = findlast(trajectory.container.sampleable_inds)
    if !isnothing(idx)
        optimise!(learner, trajectory.container[idx])
    end
end