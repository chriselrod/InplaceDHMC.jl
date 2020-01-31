"""
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that
also checks for divergence.
"""
struct TrajectoryNUTS{Tf,S}
    # "Hamiltonian."
    # H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Stepsize for leapfrog."
    ϵ::Tf
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
    "Turn statistic configuration."
    turn_statistic_configuration::S
end

function move(tree::Tree{D,T,L}, H::Hamiltonian, trajectory::TrajectoryNUTS, z::PhasePoint{D,T,L}, fwd::Bool) where {D,T,L}
    @unpack ϵ = trajectory
    leapfrog(tree, H, z, fwd ? ϵ : -ϵ)
end

###
### proposals
###

"""
$(SIGNATURES)
Random boolean which is `true` with the given probability `exp(logprob)`, which can be `≥ 1`
in which case no random value is drawn.
"""
function rand_bool_logprob(rng::AbstractRNG, logprob)
    logprob ≥ 0 || (randexp(rng, Float64) > -logprob)
end

function calculate_logprob2(::TrajectoryNUTS, is_doubling, ω₁, ω₂, ω)
    biased_progressive_logprob2(is_doubling, ω₁, ω₂, ω)
end

function combine_proposals(rng, tree::Tree, ::TrajectoryNUTS, z₁, z₂, logprob2::Real, is_forward::Bool)
    z, flag = rand_bool_logprob(rng, logprob2) ? (z₂, z₁.flag) : (z₁, z₂.flag)
    # is_doubling || free_z!(tree, flag)
    free_z!(tree, flag)
    z
end
# function combine_proposals(
    # rng, tree::Tree, ::TrajectoryNUTS, z₁, z₂, logprob2::Real, is_forward, guard₁, guard₂
# )
    # z, flag, guard = rand_bool_logprob(rng, logprob2) ? (z₂, z₁.flag, guard₁) : (z₁, z₂.flag, guard₂)
    # guard || free_z!(tree, flag)
    # z
# end

###
### statistics for visited nodes
###

struct AcceptanceStatistic{T}
    """
    Logarithm of the sum of metropolis acceptances probabilities over the whole trajectory
    (including invalid parts).
    """
    log_sum_α::T
    "Total number of leapfrog steps."
    steps::Int32
end

function combine_acceptance_statistics(A::AcceptanceStatistic, B::AcceptanceStatistic)
    AcceptanceStatistic(logaddexp(A.log_sum_α, B.log_sum_α), A.steps + B.steps)
end

"""
$(SIGNATURES)
Acceptance statistic for a leaf. The initial leaf is considered not to be visited.
"""
function leaf_acceptance_statistic(Δ, is_initial)
    is_initial ? AcceptanceStatistic(oftype(Δ, -Inf), zero(Int32)) : AcceptanceStatistic(Δ < zero(Δ) ? Δ : zero(Δ), one(Int32))
end

"""
$(SIGNATURES)
Return the acceptance rate (a `Real` betwen `0` and `1`).
"""
acceptance_rate(A::AcceptanceStatistic) = (a = exp(A.log_sum_α) / A.steps; a < one(a) ? a : one(a))

combine_visited_statistics(::TrajectoryNUTS, v, w) = combine_acceptance_statistics(v, w)

###
### turn analysis
###

"Statistics for the identification of turning points. See Betancourt (2017, appendix)."
struct GeneralizedTurnStatistic{D,T,L}
    p♯₋::FlaggedVector{D,T,L}
    p♯₊::FlaggedVector{D,T,L}
    ρ::FlaggedVector{D,T,L}
end

"""
A dummy turn statistic to use for convenient type stability; returns null ptrs that should hopefully segfault if accessed
(we'd only access them accidentally in case of a bug!)
"""
function dummy_turn_statistic(::Tree{D,T,L}) where {D,T,L}
    dummy_flag = zero(UInt32)
    dummy_ptr = reinterpret(Ptr{T}, zero(UInt))
    GeneralizedTurnStatistic(
        FlaggedVector{D,T,L}(dummy_ptr, dummy_flag),
        FlaggedVector{D,T,L}(dummy_ptr, dummy_flag),
        FlaggedVector{D,T,L}(dummy_ptr, dummy_flag)
    )
end

function leaf_turn_statistic(tree::Tree{D,T,L}, ::Val{:generalized}, H, z::PhasePoint{D,T,L}) where {D,T,L}
    p♯ = calculate_p♯(tree, H, z)
    GeneralizedTurnStatistic(p♯, p♯, FlaggedVector{D,T,L}(z.p, 0x00000000))
end

function combine_turn_statistics(
    tree::Tree, ::TrajectoryNUTS,
    x::GeneralizedTurnStatistic{D,T,L},
    y::GeneralizedTurnStatistic{D,T,L}
) where {D,T,L}
    ρₓ = x.ρ
    ρʸ = y.ρ
    # @show x.p♯₊.flag, y.p♯₋.flag
    if ρₓ.flag == 0x00000000
        # We are in depth 1; position allocated as part of a PhasePoint, not as part of a turn statistic
        # Therefore we cannot free either of them, and must allocate a new vector to store the results in.
        ρ = undefined_Σρ(tree)
        # Additionally, x.p♯₊ == x.p♯₋ and y.p♯₊ == y.p♯₋, therefore we cannot free them.
    else # we are at a depth of 2 or greater.
        # We can therefore reuse a ρ; arbitrarily, we reuse xρ
        ρ = ρₓ
        # and free yρ as well as the two discarded p♯
        free_Σρ!(tree, ρʸ.flag)
        free_ρ♯!(tree, x.p♯₊.flag)
        free_ρ♯!(tree, y.p♯₋.flag)
    end
    @inbounds @simd ivdep for l in 1:L
        ρ[l] = ρₓ[l] + ρʸ[l]
    end
    # x.p♯₊.flag == x.p♯₋.flag || free_ρ♯!(tree, x.p♯₊.flag)
    # y.p♯₊.flag == y.p♯₋.flag || free_ρ♯!(tree, y.p♯₋.flag)
    GeneralizedTurnStatistic(x.p♯₋, y.p♯₊, ρ)
end


function is_turning(::TrajectoryNUTS, τ::GeneralizedTurnStatistic{D,T,L}) where {D,T,L}
    # Uses the generalized NUTS criterion from Betancourt (2017).
    @unpack p♯₋, p♯₊, ρ = τ
    # @argcheck p♯₋ ≢ p♯₊ "internal error: is_turning called on a leaf"
    d♯₋ = zero(T)
    d♯₊ = zero(T)
    @avx for d ∈ eachindex(ρ)
        ρᵈ = ρ[d]
        d♯₋ += ρᵈ * p♯₋[d]
        d♯₊ += ρᵈ * p♯₊[d]
    end
    # Current version always calculates both dot products; alternative:
    # dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
    # Calculating both in one loop is faster
    # ( 2D fma + 3D loads vs 2D fma + 4D loads; at most 2/4 ops/cycle
    #   can be loads, so the loads cannot keep up with the 2/4 fma/cycle
    #   and are the bottleneck ),
    # but it disallows conditional checking to skip second dot product.
    #
    # Which is faster on average depends on probability of turning
    # probability of turning will be fairly low for most models.
    (d♯₋ < zero(T)) | (d♯₊ < zero(T))
end

###
### leafs
###

function leaf(tree::Tree, H::Hamiltonian, trajectory::TrajectoryNUTS, z, is_initial)
    @unpack π₀, min_Δ, turn_statistic_configuration = trajectory
    # @show logdensity(H, z), π₀
    Δ = is_initial ? zero(π₀) : logdensity(H, z) - π₀
    isdiv = Δ < min_Δ
    v = leaf_acceptance_statistic(Δ, is_initial)
    if isdiv
        # @show H
        # @show z
        # @show typeof(z)
        (z, Δ, dummy_turn_statistic(tree)), v, true
    else
        τ = leaf_turn_statistic(tree, turn_statistic_configuration, H, z)
        (z, Δ, τ), v, false
    end
end

####
#### NUTS interface
####

"""
$(TYPEDEF)
Implementation of the “No-U-Turn Sampler” of Hoffman and Gelman (2014), including subsequent
developments, as described in Betancourt (2017).
# Fields
$(FIELDS)
"""
struct NUTS{S}
    "Maximum tree depth."
    max_depth::Int
    "Threshold for negative energy relative to starting point that indicates divergence."
    min_Δ::Float64
    """
    Turn statistic configuration. Currently only `Val(:generalized)` (the default) is
    supported.
    """
    turn_statistic_configuration::S
    function NUTS(; max_depth = DEFAULT_MAX_TREE_DEPTH, min_Δ = -1000.0,
                  turn_statistic_configuration::S = Val{:generalized}()) where {S}
        # @argcheck 0 < max_depth ≤ MAX_DIRECTIONS_DEPTH
        # @argcheck min_Δ < 0
        new{S}(Int(max_depth), Float64(min_Δ), turn_statistic_configuration)
    end
end

"""
$(TYPEDEF)
Diagnostic information for a single tree built with the No-U-turn sampler.
# Fields
Accessing fields directly is part of the API.
$(FIELDS)
"""
struct TreeStatisticsNUTS # now 32 bytes; fits in ymm register
    "Log density (negative energy)."
    π::Float64
    "Acceptance rate statistic."
    acceptance_rate::Float64
    "Reason for termination. See [`InvalidTree`](@ref) and [`REACHED_MAX_DEPTH`](@ref)."
    termination::InvalidTree
    "Depth of the tree."
    depth::Int32
    "Number of leapfrog steps evaluated."
    steps::Int32
    # "Directions for tree doubling (useful for debugging)."
    # directions::Directions
end

"""
$(SIGNATURES)
No-U-turn Hamiltonian Monte Carlo transition, using Hamiltonian `H`, starting at evaluated
log density position `Q`, using stepsize `ϵ`. Parameters of `algorithm` govern the details
of tree construction.
Return two values, the new evaluated log density position, and tree statistics.
"""
function sample_tree(rng, tree::Tree, algorithm::NUTS, H::Hamiltonian, z::PhasePoint, ϵ;
                          p = nothing, directions = rand(rng, Directions))
    if p === nothing
        rand_p!(rng, z.p, H.κ)
    else
        @unpack Q, flag = z
        z = PhasePoint(Q, p, flag)
    end
    @unpack max_depth, min_Δ, turn_statistic_configuration = algorithm
    trajectory = TrajectoryNUTS(logdensity(H, z), ϵ, min_Δ, turn_statistic_configuration)
    ζ, v, termination, depth = sample_trajectory(rng, tree, H, trajectory, z, max_depth, directions)
    tree_statistics = TreeStatisticsNUTS(logdensity(H, ζ), acceptance_rate(v), termination, depth, v.steps)#, directions)
    ζ, tree_statistics
end



