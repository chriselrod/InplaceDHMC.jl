module InplaceDHMC

export
    # kinetic energy
    GaussianKineticEnergy,
    # NUTS
    TreeOptionsNUTS,
    # reporting
    NoProgressReport, LogProgressReport,
    # mcmc
    InitialStepsizeSearch, DualAveraging, FindLocalOptimum,
    TuningNUTS, mcmc_with_warmup, default_warmup_stages, fixed_stepsize_warmup_stages

# using ArgCheck: @argcheck
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using LinearAlgebra
using LinearAlgebra: Diagonal, Symmetric
# using LogDensityProblems: capabilities, LogDensityOrder, dimension, logdensity_and_gradient
# import NLSolversBase, Optim # optimization step in mcmc
using Parameters: @unpack
using Random: AbstractRNG, randn, Random, randexp
using Statistics: cov, mean, median, middle, quantile, var

using VectorizationBase, LoopVectorization, StackPointers, PaddedMatrices, QuasiNewtonMethods
using ProbabilityModels: AbstractProbabilityModel

# copy from StatsFuns.jl
function logaddexp(x, y)
    isfinite(x) && isfinite(y) || return x > y ? x : y # QuasiNewtonMethods.nanmax(x,y)
    x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
end





#####
##### Abstract tree/trajectory interface
#####

####
#### Directions
####

"Maximum number of iterations [`next_direction`](@ref) supports."
const MAX_DIRECTIONS_DEPTH = 64

"""
Internal type implementing random directions. Draw a new value with `rand`, see
[`next_direction`](@ref).
Serves two purposes: a fixed value of `Directions` is useful for unit testing, and drawing a
single bit flag collection economizes on the RNG cost.
"""
struct Directions
    flags::UInt64
end

Base.rand(rng::AbstractRNG, ::Type{Directions}) = Directions(rand(rng, UInt64))

"""
$(SIGNATURES)
Return the next direction flag and the new state of directions. Results are undefined for
more than [`MAX_DIRECTIONS_DEPTH`](@ref) updates.
"""
function next_direction(directions::Directions)
    @unpack flags = directions
    Bool(flags & 0x01), Directions(flags >>> 1)
end

####
#### Trajectory interface
####

"""
    $(FUNCTIONNAME)(trajectory, z, is_forward)
Move along the trajectory in the specified direction. Return the new position.
"""
function move end

"""
    $(FUNCTIONNAME)(trajectory, τ)
Test if the turn statistics `τ` indicate that the corresponding tree is turning.
Will only be called on nontrivial trees (at least two nodes).
"""
function is_turning end

"""
    $(FUNCTIONNAME)(trajectory, τ₁, τ₂)
Combine turn statistics on trajectory. Implementation can assume that the trees that
correspond to the turn statistics have the same ordering.
"""
function combine_turn_statistics end

"""
    $(FUNCTIONNAME)(trajectory, v₁, v₂)
Combine visited node statistics for adjacent trees trajectory. Implementation should be
invariant to the ordering of `v₁` and `v₂` (ie the operation is commutative).
"""
function combine_visited_statistics end

"""
    $(FUNCTIONNAME)(trajectory, is_doubling::Bool, ω₁, ω₂, ω)
Calculate the log probability if selecting the subtree corresponding to `ω₂`. Being the log
of a probability, it is always `≤ 0`, but implementations are allowed to return and accept
values `> 0` and treat them as `0`.
When `is_doubling`, the tree corresponding to `ω₂` was obtained from a doubling step (this
can be relevant eg for biased progressive sampling).
The value `ω = logaddexp(ω₁, ω₂)` is provided for avoiding redundant calculations.
See [`biased_progressive_logprob2`](@ref) for an implementation.
"""
function calculate_logprob2 end

"""
    $(FUNCTIONNAME)(rng, trajectory, ζ₁, ζ₂, logprob2::Real, is_forward::Bool)
Combine two proposals `ζ₁, ζ₂` on `trajectory`, with log probability `logprob2` for
selecting `ζ₂`.
 `ζ₁` is before `ζ₂` iff `is_forward`.
"""
function combine_proposals end

"""
    ζωτ_or_nothing, v = $(FUNCTIONNAME)(trajectory, z, is_initial)
Information for a tree made of a single node. When `is_initial == true`, this is the first
node.
The first value is either
1. `nothing` for a divergent node,
2. a tuple containing the proposal `ζ`, the log weight (probability) of the node `ω`, the
turn statistics `τ` (never tested as with `is_turning` for leafs).
The second value is the visited node information.
"""
function leaf end

####
#### utilities
####

"""
$(SIGNATURES)
Combine turn statistics with the given direction. When `is_forward`, `τ₁` is before `τ₂`,
otherwise after.
Internal helper function.
"""
function combine_turn_statistics_in_direction(trajectory, τ₁, τ₂, is_forward::Bool)
    if is_forward
        combine_turn_statistics(trajectory, τ₁, τ₂)
    else
        combine_turn_statistics(trajectory, τ₂, τ₁)
    end
end

function combine_proposals_and_logweights(rng, trajectory, ζ₁, ζ₂, ω₁::Real, ω₂::Real,
                                          is_forward::Bool, is_doubling::Bool)
    ω = logaddexp(ω₁, ω₂)
    logprob2 = calculate_logprob2(trajectory, is_doubling, ω₁, ω₂, ω)
    ζ = combine_proposals(rng, trajectory, ζ₁, ζ₂, logprob2, is_forward)
    ζ, ω
end

"""
$(SIGNATURES)
Given (relative) log probabilities `ω₁` and `ω₂`, return the log probabiliy of
drawing a sample from the second (`logprob2`).
When `bias`, biases towards the second argument, introducing anti-correlations.
"""
function biased_progressive_logprob2(bias::Bool, ω₁::Real, ω₂::Real, ω = logaddexp(ω₁, ω₂))
    ω₂ - (bias ? ω₁ : ω)
end

####
#### abstract trajectory interface
####

"""
$(SIGNATURES)
Information about an invalid (sub)tree, using positions relative to the starting node.
1. When `left < right`, this tree was *turning*.
2. When `left == right`, this is a *divergent* node.
3. `left == 1 && right == 0` is used as a sentinel value for reaching maximum depth without
encountering any invalid trees (see [`REACHED_MAX_DEPTH`](@ref). All other `left > right`
values are disallowed.
"""
struct InvalidTree
    left::Int
    right::Int
end

InvalidTree(i::Integer) = InvalidTree(i, i)

is_divergent(invalid_tree::InvalidTree) = invalid_tree.left == invalid_tree.right

function Base.show(io::IO, invalid_tree::InvalidTree)
    msg = if is_divergent(invalid_tree)
        "divergence at position $(invalid_tree.left)"
    elseif invalid_tree == REACHED_MAX_DEPTH
        "reached maximum depth without divergence or turning"
    else
        @unpack left, right = invalid_tree
        "turning at positions $(left):$(right)"
    end
    print(io, msg)
end

"Sentinel value for reaching maximum depth."
const REACHED_MAX_DEPTH = InvalidTree(1, 0)

# topptr(a::Ptr{T}, b::Ptr{T}) where {T} = reinterpret(Ptr{T}, max(reinterpret(UInt, a), reinterpret(UInt, b)))

"""
    result, v = adjacent_tree(rng, trajectory, z, i, depth, is_forward)
Traverse the tree of given `depth` adjacent to point `z` in `trajectory`.
`is_forward` specifies the direction, `rng` is used for random numbers in
[`combine_proposals`](@ref). `i` is an integer position relative to the initial node (`0`).
The *first value* is either
1. an `InvalidTree`, indicating the first divergent node or turning subtree that was
encounteted and invalidated this tree.
2. a tuple of `(ζ, ω, τ, z′, i′), with
    - `ζ`: the proposal from the tree.
    - `ω`: the log weight of the subtree that corresponds to the proposal
    - `τ`: turn statistics
    - `z′`: the last node of the tree
    - `i′`: the position of the last node relative to the initial node.
The *second value* is always the visited node statistic.
"""
function adjacent_tree(rng, sptr::StackPointer, trajectory, z::, i, depth, is_forward) where {P,T,L}
    i′ = i + (is_forward ? 1 : -1)
    if depth == 0 # moves from z into ζready
        z′ = move(sptr, trajectory, z, is_forward)
        ζωτ, v, invalid = leaf(trajectory, z′, false)
        return (ζωτ..., z′, i′), v, (invalid,InvalidTree(i′))
    else
        # “left” tree
        t₋, v₋,(invalid,it) = adjacent_tree(rng, sptr, trajectory, z, i, depth - 1, is_forward)
        invalid && return t₋, v₋, (invalid, it)
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        # “right” tree — visited information from left is kept even if invalid
        sptr_right = StackPointer(pointer(ζ₋) + sizeof(T)*2L)
        t₊, v₊,(invalid,it) = adjacent_tree(rng, sptr_right, trajectory, z₋, i₋, depth - 1, is_forward)
        v = combine_visited_statistics(trajectory, v₋, v₊)
        invalid && return t₊, v,(invalid,it)
        ζ₊, ω₊, τ₊, z₊, i₊ = t₊

        # turning invalidates
        τ = combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        is_turning(trajectory, τ) && return t₊, v, (true, InvalidTree(i′, i₊))

        # valid subtree, combine proposals
        ζ, ζempty, ω = combine_proposals_and_logweights(rng, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false)
        (ζ, ω, τ, z₊, i₊), v, (false,REACHED_MAX_DEPTH)
    end
end


"""
$(SIGNATURES)
Sample a `trajectory` starting at `z`, up to `max_depth`. `directions` determines the tree
expansion directions.
Return the following values
- `ζ`: proposal from the tree
- `v`: visited node statistics
- `termination`: an `InvalidTree` (this includes the last doubling step turning, which is
  technically a valid tree) or `REACHED_MAX_DEPTH` when all subtrees were valid and no
  turning happens.
- `depth`: the depth of the tree that was sampled from. Doubling steps that lead to an
  invalid adjacent tree do not contribute to `depth`.
"""
function sample_trajectory(rng, trajectory, z::PtrVector{P,T,L,L}, max_depth::Integer, directions::Directions) where {P,T,L,L}
    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    (ζ, ω, τ), v, invalid = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = REACHED_MAX_DEPTH
    i₋ = i₊ = 0
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        t′, v′, (invalid, it) = adjacent_tree(
            StackPointer(pointer(ζ) + sizeof(T)*2L), rng, trajectory, (is_forward ? z₊ : z₋), (is_forward ? i₊ : i₋), depth, is_forward
        )
        v = combine_visited_statistics(trajectory, v, v′)

        # invalid adjacent tree: stop
        invalid && (termination = it; break)

        # extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′

        # update edges and combine proposals
        if is_forward
            z₊, i₊ = z′, i′
        else
            z₋, i₋ = z′, i′
        end

        # tree has doubled successfully
        ζ, ω = combine_proposals_and_logweights(rng, trajectory, ζ, ζ′, ω, ω′, is_forward, true)
        depth += 1

        # when the combined tree is turning, stop
        τ = combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        is_turning(trajectory, τ) && (termination = InvalidTree(i₋, i₊); break)
    end
    ζ, v, termination, depth
end

# include("trees.jl")


#####
##### Building blocks for traversing a Hamiltonian deterministically, using the leapfrog
##### integrator.
#####

####
#### kinetic energy
####

"""
$(TYPEDEF)
Kinetic energy specifications. Implements the methods
- `Base.size`
- [`kinetic_energy`](@ref)
- [`calculate_p♯`](@ref)
- [`∇kinetic_energy`](@ref)
- [`rand_p`](@ref)
For all subtypes, it is implicitly assumed that kinetic energy is symmetric in
the momentum `p`,
```julia
kinetic_energy(κ, p, q) == kinetic_energy(κ, .-p, q)
```
When the above is violated, the consequences are undefined.
"""
abstract type KineticEnergy end

"""
$(TYPEDEF)
Euclidean kinetic energies (position independent).
"""
abstract type EuclideanKineticEnergy <: KineticEnergy end

"""
$(TYPEDEF)
Gaussian kinetic energy, with ``K(q,p) = p ∣ q ∼ 1/2 pᵀ⋅M⁻¹⋅p + log|M|`` (without constant),
which is independent of ``q``.
The inverse covariance ``M⁻¹`` is stored.
!!! note
    Making ``M⁻¹`` approximate the posterior variance is a reasonable starting point.
"""
struct GaussianKineticEnergy{P,T,L} <: EuclideanKineticEnergy
    "M⁻¹"
    M⁻¹::Diagonal{T,PtrVector{P,T,L,L,true}}
    "W such that W*W'=M. Used for generating random draws."
    W::Diagonal{T,PtrVector{P,T,L,L,true}}
end

# """
# $(SIGNATURES)
# Gaussian kinetic energy with the given inverse covariance matrix `M⁻¹`.
# """
# GaussianKineticEnergy(M⁻¹::AbstractMatrix) = GaussianKineticEnergy(M⁻¹, cholesky(inv(M⁻¹)).L)

"""
$(SIGNATURES)
Gaussian kinetic energy with the given inverse covariance matrix `M⁻¹`.
"""
function GaussianKineticEnergy(sptr::StackPointer, M⁻¹::Diagonal{T,PtrVector{P,T,L,L,true}}) where {P,T,L}
    sptr, W = PtrVector{P,T,L,L,true}(sptr)
    M⁻¹d = M⁻¹.diag
    @fastmath @inbounds @simd for l ∈ 1:L
        W[l] = one(T) / sqrt( M⁻¹d[l] )
    end
    sptr, GaussianKineticEnergy(M⁻¹, Diagonal( W ))
end

"""
$(SIGNATURES)
Gaussian kinetic energy with a diagonal inverse covariance matrix `M⁻¹=m⁻¹*I`.
"""
function GaussianKineticEnergy(sptr::StackPointer, ::Val{N}, m⁻¹::T = 1.0) where {N,T}
    sptr, M⁻¹ = PtrVector{N,T}(sptr)
    fill!(M⁻¹, m⁻¹)
    GaussianKineticEnergy(sptr, Diagonal(M⁻¹))
end

@generated function GaussianKineticEnergy(sp::StackPointer, sample::AbstractMatrix{T}, λ::T, ::Val{D}) where {D,T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W-1
    M = (D + Wm1) & ~Wm1
    V = Vec{W,T}
    # note that defining M as we did means Wrem == 0
    MdW = (M + Wm1) >> W
    Wrem = M & Wm1
    size_T = sizeof(T)
    WT = size_T*W
    need_to_mask = Wrem > 0
    reps_per_block = 4
    blocked_reps, blocked_rem = divrem(MdW, reps_per_block)
    if (MdW % (blocked_reps + 1)) == 0
        blocked_reps, blocked_rem = blocked_reps + 1, 0
        reps_per_block = MdW ÷ blocked_reps
    end
    if (need_to_mask && blocked_rem == 0) || blocked_reps == 1
        blocked_rem += reps_per_block
        blocked_reps -= 1
    end
    AL = VectorizationBase.align(M*size_T)
    q = quote
        ptr_s² = pointer(sp, $T)
        regs² = PtrVector{$M,$T}(ptr_s²)
        ptr_invs = ptr_s² + $AL
        invs = PtrVector{$M,$T}(ptr_invs)

        N = size(sample, 2)
        ptr_sample = pointer(sample)
        @fastmath begin
            Ninv = one($T) / N
            mulreg = N / ((N + λ)*(N - 1))
            addreg = $(T(1e-3)) * λ / (N + λ)
        end
        vNinv = SIMDPirates.vbroadcast($V, Ninv)
        vmulreg = SIMDPirates.vbroadcast($V, mulreg)
        vaddreg = SIMDPirates.vbroadcast($V, addreg)        
    end
    if blocked_reps > 
        loop_block = regularized_cov_block_quote(W, T, reps_per_block, M, false)
        block_rep_quote = quote
            for _ ∈ 1:$blocked_reps
                $loop_block
                ptr_sample += $WT*$reps_per_block
                ptr_s² += $WT*$reps_per_block
                ptr_invs += $WT*$reps_per_block
            end
        end
        push!(q.args, block_rep_quote)
    end
    if blocked_rem > 0
        push!(q.args, regularized_cov_block_quote(W, T, blocked_rem, M, need_to_mask, VectorizationBase.mask(T,Wrem)))        
    end
    push!(q.args, :(sp + $(2AL), GaussianKineticEnergy(Diagonal(regs²), Diagonal(invs))))
    q
end


function Base.show(io::IO, κ::GaussianKineticEnergy{T}) where {T}
    print(io::IO, "Gaussian kinetic energy ($(Base.typename(T))), √diag(M⁻¹): $(.√(diag(κ.M⁻¹)))")
end

## NOTE about implementation: the 3 methods are callable without a third argument (`q`)
## because they are defined for Gaussian (Euclidean) kinetic energies.

Base.size(κ::GaussianKineticEnergy, args...) = size(κ.M⁻¹, args...)

"""
$(SIGNATURES)
Return kinetic energy `κ`, at momentum `p`.
"""
@generated function kinetic_energy(κ::GaussianKineticEnergy{P,T,L}, p::PtrVector{P,T,L}, q = nothing) where {P,T,L}
    quote
        M⁻¹ = κ.M⁻¹.diag
        ke = zero(T)
        @vvectorize for m ∈ 1:$P
            pₘ = p[m]
            ke += pₘ * M⁻¹[m] * pₘ
        end
        T(0.5) * ke
    end
end
# kinetic_energy(κ::GaussianKineticEnergy, p, q = nothing) = dot(p, κ.M⁻¹ * p) / 2

"""
$(SIGNATURES)
Return ``p♯ = M⁻¹⋅p``, used for turn diagnostics.
"""
function calculate_p♯(sptr::StackPointer, κ::GaussianKineticEnergy, p::PtrVector{P,T,L}, q = nothing)
    M⁻¹ = κ.M⁻¹
    sptr, M⁻¹p = PtrVector{P,T,L,L}(sptr)
    @inbounds @simd for l ∈ 1:L
        M⁻¹p[l] = M⁻¹[l] * p[l]
    end
    sptr, M⁻¹p
end

"""
$(SIGNATURES)
Calculate the gradient of the logarithm of kinetic energy in momentum `p`.
"""
∇kinetic_energy(sptr::StackPointer, κ::GaussianKineticEnergy, p, q = nothing) = calculate_p♯(sptr, κ, p)

"""
$(SIGNATURES)
Generate a random momentum from a kinetic energy at position `q`.
"""
function rand_p(sptr::StackPointer, rng::AbstractRNG, κ::GaussianKineticEnergy{P,T,L}, q = nothing) where {P,T,L}
    W = κ.W
    sptr, r = PtrVector{P,T,L}(sptr)
    sptr, randn!(rng, r, W)
end

####
#### Hamiltonian
####

struct Hamiltonian{K,L}
    "The kinetic energy specification."
    κ::K
    """
    The (log) density we are sampling from. Supports the `LogDensityProblem` API.
    Technically, it is the negative of the potential energy.
    """
    ℓ::L
    """
    $(SIGNATURES)
    Construct a Hamiltonian from the log density `ℓ`, and the kinetic energy specification
    `κ`. `ℓ` with a vector are expected to support the `LogDensityProblems` API, with
    gradients.
    """
    function Hamiltonian(κ::K, ℓ::L) where {K <: GaussianKineticEnergy{P,T},L<:AbstractProbabilityModel{P}}
        # @argcheck capabilities(ℓ) ≥ LogDensityOrder(1)
        # @argcheck dimension(ℓ) == size(κ, 1)
        new{K,L}(κ, ℓ)
    end
end

Base.show(io::IO, H::Hamiltonian) = print(io, "Hamiltonian with $(H.κ)")

"""
$(TYPEDEF)
A log density evaluated at position `q`. The log densities and gradient are saved, so that
they are not calculated twice for every leapfrog step (both as start- and endpoints).
Because of caching, a `EvaluatedLogDensity` should only be used with a specific Hamiltonian,
preferably constructed with the `evaluate_ℓ` constructor.
In composite types and arguments, `Q` is usually used for this type.
"""
struct EvaluatedLogDensity{P,T,L}
    "Position."
    q::PtrVector{P,T,L,L,true}
    "ℓ(q). Saved for reuse in sampling."
    ℓq::T
    "∇ℓ(q). Cached for reuse in sampling."
    ∇ℓq::PtrVector{P,T,L,L,true}
    function EvaluatedLogDensity(q::PtrVector{P,T,L,L}, ℓq::T, ∇ℓq::PtrVector{P,T,L,L}) where {P,T,L}
#        @argcheck length(q) == length(∇ℓq)
        new{P,T,L}(q, ℓq, ∇ℓq)
    end
end

# general constructors below are necessary to sanitize input from eg Diagnostics, or an
# initial position given as integers, etc

function EvaluatedLogDensity(q::AbstractVector, ℓq::Real, ∇ℓq::AbstractVector)
    q, ∇ℓq = promote(q, ∇ℓq)
    EvaluatedLogDensity(q, ℓq, ∇ℓq)
end

EvaluatedLogDensity(q, ℓq::Real, ∇ℓq) = EvaluatedLogDensity(collect(q), ℓq, collect(∇ℓq))

"""
$(SIGNATURES)
Evaluate log density and gradient and save with the position. Preferred interface for
creating `EvaluatedLogDensity` instances.
"""
function evaluate_ℓ(sptr::StackPointer, ℓ::AbstractProbabilityModel{P}, q::PtrVector{P,T,L,L}) where {P,T,L}
    sptr2, ∇ℓq = PtrVector{P,T,L,L}(sptr)
    ℓq = logdensity_and_gradient!(∇ℓq, ℓ, q, sptr2)
    if isfinite(ℓq)
        sptr2, EvaluatedLogDensity(q, ℓq, ∇ℓq)
    else
        sptr, EvaluatedLogDensity(q, oftype(ℓq, -Inf), q) # second q used just as a placeholder
    end
end

"""
$(TYPEDEF)
A point in phase space, consists of a position (in the form of an evaluated log density `ℓ`
at `q`) and a momentum.
"""
struct PhasePoint{P,T,L}
    "Evaluated log density."
    Q::EvaluatedLogDensity{P,T,L}
    "Momentum."
    p::PtrVector{P,T,L,L,true}
    # function PhasePoint(Q::EvaluatedLogDensity, p::S) where {T,S}
        # @argcheck length(p) == length(Q.q)
        # new{T,S}(Q, p)
    # end
end

"""
$(SIGNATURES)
Log density for Hamiltonian `H` at point `z`.
If `ℓ(q) == -Inf` (rejected), skips the kinetic energy calculation.
Non-finite values (incl `NaN`, `Inf`) are automatically converted to `-Inf`. This can happen
if
1. the log density is not a finite value,
2. the kinetic energy is not a finite value (which usually happens when `NaN` or `Inf` got
mixed in the leapfrog step, leading to an invalid position).
"""
function logdensity(H::Hamiltonian{<:EuclideanKineticEnergy}, z::PhasePoint)
    @unpack ℓq = z.Q
    isfinite(ℓq) || return oftype(ℓq, -Inf)
    K = kinetic_energy(H.κ, z.p)
    ℓq - (isfinite(K) ? K : oftype(K, Inf))
end

function calculate_p♯(H::Hamiltonian{<:EuclideanKineticEnergy}, z::PhasePoint)
    calculate_p♯(H.κ, z.p)
end

"""
    leapfrog(H, z, ϵ)
Take a leapfrog step of length `ϵ` from `z` along the Hamiltonian `H`.
Return the new phase point.
The leapfrog algorithm uses the gradient of the next position to evolve the momentum. If
this is not finite, the momentum won't be either, `logdensity` above will catch this and
return an `-Inf`, making the point divergent.
"""
function leapfrog(sp::StackPointer,
        H::Hamiltonian{GaussianKineticEnergy{<:Diagonal}},
        z::PhasePoint{EvaluatedLogDensity{P,T,L}}, ϵ
) where {P,L,T}
    @unpack ℓ, κ = H
    @unpack p, Q = z
#    @argcheck isfinite(Q.ℓq) "Internal error: leapfrog called from non-finite log density"
    sptr = pointer(sp, T)
    B = L*sizeof(T) # counting on this being aligned.
    pₘ = PtrVector{P,T,L,L}(sptr)
    q′ = PtrVector{P,T,L,L}(sptr + B) # should this be a new Vector?
    M⁻¹ = κ.M⁻¹.diag
    ∇ℓq = Q.∇ℓq
    q = Q.q
    @fastmath @inbounds @simd for l ∈ 1:L
        pₘₗ = p[l] + ϵₕ * ∇ℓq[l]
        pₘ[l] = pₘₗ
        q′[l] = q[l] + ϵ * M⁻¹[l] * pₘₗ
    end
    # Variables that escape:
    # p′, Q′ (q′, ∇ℓq)
    sp, Q′ = evaluate_ℓ(sp + 2B, H.ℓ, q′)
    ∇ℓq′ = Q′.∇ℓq
    p′ = pₘ # PtrVector{P,T,L,L}(sptr + 3bytes)
    @fastmath @inbounds @simd for l ∈ 1:L
        p′[l] = pₘ[l] + ϵₕ * ∇ℓq′[l]
    end
    sp + 3B, DynamicHMC.PhasePoint(Q′, p′)
end
# function DynamicHMC.move(sp::StackPointer, trajectory::DynamicHMC.TrajectoryNUTS, z, fwd)
    # @unpack H, ϵ = trajectory
    # leapfrog(sp, H, z, fwd ? ϵ : -ϵ)
# end


# include("hamiltonian.jl")


#####
##### stepsize heuristics and adaptation
#####

####
#### initial stepsize
####

"""
$(TYPEDEF)
Parameters for the search algorithm for the initial stepsize.
The algorithm finds an initial stepsize ``ϵ`` so that the local acceptance ratio
``A(ϵ)`` satisfies
```math
a_\\text{min} ≤ A(ϵ) ≤ a_\\text{max}
```
This is achieved by an initial bracketing, then bisection.
$FIELDS
!!! note
    Cf. Hoffman and Gelman (2014), which does not ensure bounds for the
    acceptance ratio, just that it has crossed a threshold. This version seems
    to work better for some tricky posteriors with high curvature.
"""
struct InitialStepsizeSearch
    "Lowest local acceptance rate."
    a_min::Float64
    "Highest local acceptance rate."
    a_max::Float64
    "Initial stepsize."
    ϵ₀::Float64
    "Scale factor for initial bracketing, > 1. *Default*: `2.0`."
    C::Float64
    "Maximum number of iterations for initial bracketing."
    maxiter_crossing::Int
    "Maximum number of iterations for bisection."
    maxiter_bisect::Int
    function InitialStepsizeSearch(; a_min = 0.25, a_max = 0.75, ϵ₀ = 1.0, C = 2.0,
                                   maxiter_crossing = 400, maxiter_bisect = 400)
        @argcheck 0 < a_min < a_max < 1
        @argcheck 0 < ϵ₀
        @argcheck 1 < C
        @argcheck maxiter_crossing ≥ 50
        @argcheck maxiter_bisect ≥ 50
        new(a_min, a_max, ϵ₀, C, maxiter_crossing, maxiter_bisect)
    end
end

"""
$(SIGNATURES)
Find the stepsize for which the local acceptance rate `A(ϵ)` crosses `a`.
Return `ϵ₀, A(ϵ₀), ϵ₁`, A(ϵ₁)`, where `ϵ₀` and `ϵ₁` are stepsizes before and
after crossing `a` with `A(ϵ)`, respectively.
Assumes that ``A(ϵ₀) ∉ (a_\\text{min}, a_\\text{max})``, where the latter are
defined in `parameters`.
- `parameters`: parameters for the iteration.
- `A`: local acceptance ratio (uncapped), a function of stepsize `ϵ`
- `ϵ₀`, `Aϵ₀`: initial value of `ϵ`, and `A(ϵ₀)`
"""
function find_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀ = A(ϵ₀))
    @unpack a_min, a_max, C, maxiter_crossing = parameters
    s, a = Aϵ₀ > a_max ? (1.0, a_max) : (-1.0, a_min)
    if s < 0                    # when A(ϵ) < a,
        C = 1/C                 # decrease ϵ
    end
    for _ in 1:maxiter_crossing
        ϵ = ϵ₀ * C
        Aϵ = A(ϵ)
        if s*(Aϵ - a) ≤ 0
            return ϵ₀, Aϵ₀, ϵ, Aϵ
        else
            ϵ₀ = ϵ
            Aϵ₀ = Aϵ
        end
    end
    # should never each this, miscoded log density?
    dir = s > 0 ? "below" : "above"
    error("Reached maximum number of iterations searching for ϵ from $(dir).")
end

"""
$(SIGNATURES)
Return the desired stepsize `ϵ` by bisection.
- `parameters`: algorithm parameters, see [`InitialStepsizeSearch`](@ref)
- `A`: local acceptance ratio (uncapped), a function of stepsize `ϵ`
- `ϵ₀`, `ϵ₁`, `Aϵ₀`, `Aϵ₁`: stepsizes and acceptance rates (latter optional).
This function assumes that ``ϵ₀ < ϵ₁``, the stepsize is not yet acceptable, and
the cached `A` values have the correct ordering.
"""
function bisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀ = A(ϵ₀), Aϵ₁ = A(ϵ₁))
    @unpack a_min, a_max, maxiter_bisect = parameters
    @argcheck ϵ₀ < ϵ₁
    @argcheck Aϵ₀ > a_max && Aϵ₁ < a_min
    for _ in 1:maxiter_bisect
        ϵₘ = middle(ϵ₀, ϵ₁)
        Aϵₘ = A(ϵₘ)
        if a_min ≤ Aϵₘ ≤ a_max  # in
            return ϵₘ
        elseif Aϵₘ < a_min      # above
            ϵ₁ = ϵₘ
            Aϵ₁ = Aϵₘ
        else                    # below
            ϵ₀ = ϵₘ
            Aϵ₀ = Aϵₘ
        end
    end
    # should never each this, miscoded log density?
    error("Reached maximum number of iterations while bisecting interval for ϵ.")
end

"""
$(SIGNATURES)
Find an initial stepsize that matches the conditions of `parameters` (see
[`InitialStepsizeSearch`](@ref)).
`A` is the local acceptance ratio (uncapped). When given a Hamiltonian `H` and a
phasepoint `z`, it will be calculated using [`local_acceptance_ratio`](@ref).
"""
function find_initial_stepsize(parameters::InitialStepsizeSearch, A)
    @unpack a_min, a_max, ϵ₀ = parameters
    Aϵ₀ = A(ϵ₀)
    if a_min ≤ Aϵ₀ ≤ a_max
        ϵ₀
    else
        ϵ₀, Aϵ₀, ϵ₁, Aϵ₁ = find_crossing_stepsize(parameters, A, ϵ₀, Aϵ₀)
        if a_min ≤ Aϵ₁ ≤ a_max  # in interval
            ϵ₁
        elseif ϵ₀ < ϵ₁          # order as necessary
            bisect_stepsize(parameters, A, ϵ₀, ϵ₁, Aϵ₀, Aϵ₁)
        else
            bisect_stepsize(parameters, A, ϵ₁, ϵ₀, Aϵ₁, Aϵ₀)
        end
    end
end

@noinline ThrowDomainError(args...) = thow(DomainError(args...))

"""
$(SIGNATURES)
Uncapped log acceptance ratio of a Langevin step.
"""
function log_acceptance_ratio(sptr, H, z, ϵ)
    target = logdensity(H, z)
    isfinite(target) || ThrowDomainError(z, "Starting point has non-finite density.")
    logdensity(H, leapfrog(sptr, H, z, ϵ), sptr) - target
end

"""
$(SIGNATURES)
Return a function of the stepsize (``ϵ``) that calculates the local acceptance
ratio for a single leapfrog step around `z` along the Hamiltonian `H`. Formally,
let
```julia
A(ϵ) = exp(logdensity(H, leapfrog(H, z, ϵ)) - logdensity(H, z))
```
Note that the ratio is not capped by `1`, so it is not a valid probability *per se*.
"""
function local_acceptance_ratio(H, z)
    target = logdensity(H, z)
    isfinite(target) ||
        throw(DomainError(z.p, "Starting point has non-finite density."))
    ϵ -> exp(logdensity(H, leapfrog(H, z, ϵ)) - target)
end

function find_initial_stepsize(parameters::InitialStepsizeSearch, H, z)
    find_initial_stepsize(parameters, local_acceptance_ratio(H, z))
end

"""
$(TYPEDEF)
Parameters for the dual averaging algorithm of Gelman and Hoffman (2014, Algorithm 6).
To get reasonable defaults, initialize with `DualAveraging()`.
# Fields
$(FIELDS)
"""
struct DualAveraging{T}
    "target acceptance rate"
    δ::T
    "regularization scale"
    γ::T
    "relaxation exponent"
    κ::T
    "offset"
    t₀::Int
    function DualAveraging(δ::T, γ::T, κ::T, t₀::Int) where {T <: Real}
        @argcheck 0 < δ < 1
        @argcheck γ > 0
        @argcheck 0.5 < κ ≤ 1
        @argcheck t₀ ≥ 0
        new{T}(δ, γ, κ, t₀)
    end
end

function DualAveraging(; δ = 0.8, γ = 0.05, κ = 0.75, t₀ = 10)
    DualAveraging(promote(δ, γ, κ)..., t₀)
end

"Current state of adaptation for `ϵ`."
struct DualAveragingState{T <: AbstractFloat}
    μ::T
    m::Int
    H̄::T
    logϵ::T
    logϵ̄::T
end

"""
$(SIGNATURES)
Return an initial adaptation state for the adaptation method and a stepsize `ϵ`.
"""
function initial_adaptation_state(::DualAveraging, ϵ)
    @argcheck ϵ > 0
    logϵ = log(ϵ)
    DualAveragingState(log(10) + logϵ, 0, zero(logϵ), logϵ, zero(logϵ))
end

"""
$(SIGNATURES)
Update the adaptation `A` of log stepsize `logϵ` with average Metropolis acceptance rate `a`
over the whole visited trajectory, using the dual averaging algorithm of Gelman and Hoffman
(2014, Algorithm 6). Return the new adaptation state.
"""
function adapt_stepsize(parameters::DualAveraging, A::DualAveragingState, a)
    @argcheck 0 ≤ a ≤ 1
    @unpack δ, γ, κ, t₀ = parameters
    @unpack μ, m, H̄, logϵ, logϵ̄ = A
    m += 1
    H̄ += (δ - a - H̄) / (m + t₀)
    logϵ = μ - √m/γ * H̄
    logϵ̄ += m^(-κ)*(logϵ - logϵ̄)
    DualAveragingState(μ, m, H̄, logϵ, logϵ̄)
end

"""
$(SIGNATURES)
Return the stepsize `ϵ` for the next HMC step while adapting.
"""
current_ϵ(A::DualAveragingState, tuning = true) = exp(A.logϵ)

"""
$(SIGNATURES)
Return the final stepsize `ϵ` after adaptation.
"""
final_ϵ(A::DualAveragingState, tuning = true) = exp(A.logϵ̄)

###
### fixed stepsize adaptation placeholder
###

"""
$(SIGNATURES)
Adaptation with fixed stepsize. Leaves `ϵ` unchanged.
"""
struct FixedStepsize end

initial_adaptation_state(::FixedStepsize, ϵ) = ϵ

adapt_stepsize(::FixedStepsize, ϵ, a) = ϵ

current_ϵ(ϵ::Real) = ϵ

final_ϵ(ϵ::Real) = ϵ


# include("stepsize.jl")

#####
##### NUTS tree sampler implementation.
#####

####
#### Trajectory and implementation
####

"""
Representation of a trajectory, ie a Hamiltonian with a discrete integrator that
also checks for divergence.
"""
struct TrajectoryNUTS{TH,Tf,S}
    "Hamiltonian."
    H::TH
    "Log density of z (negative log energy) at initial point."
    π₀::Tf
    "Stepsize for leapfrog."
    ϵ::Tf
    "Smallest decrease allowed in the log density."
    min_Δ::Tf
    "Turn statistic configuration."
    turn_statistic_configuration::S
end

function move(sptr, trajectory::TrajectoryNUTS, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(sptr, H, z, fwd ? ϵ : -ϵ)
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

function combine_proposals(rng, ::TrajectoryNUTS, z₁, z₂, logprob2::Real, is_forward)
    rand_bool_logprob(rng, logprob2) ? z₂ : z₁
end

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
    steps::Int
end

function combine_acceptance_statistics(A::AcceptanceStatistic, B::AcceptanceStatistic)
    AcceptanceStatistic(logaddexp(A.log_sum_α, B.log_sum_α), A.steps + B.steps)
end

"""
$(SIGNATURES)
Acceptance statistic for a leaf. The initial leaf is considered not to be visited.
"""
function leaf_acceptance_statistic(Δ, is_initial)
    is_initial ? AcceptanceStatistic(oftype(Δ, -Inf), 0) : AcceptanceStatistic(min(Δ, 0), 1)
end

"""
$(SIGNATURES)
Return the acceptance rate (a `Real` betwen `0` and `1`).
"""
acceptance_rate(A::AcceptanceStatistic) = min(exp(A.log_sum_α) / A.steps, 1)

combine_visited_statistics(::TrajectoryNUTS, v, w) = combine_acceptance_statistics(v, w)

###
### turn analysis
###

"Statistics for the identification of turning points. See Betancourt (2017, appendix)."
struct GeneralizedTurnStatistic{T}
    p♯₋::T
    p♯₊::T
    ρ::T
end

function leaf_turn_statistic(::Val{:generalized}, H, z)
    p♯ = calculate_p♯(H, z)
    GeneralizedTurnStatistic(p♯, p♯, z.p)
end

function combine_turn_statistics(::TrajectoryNUTS,
                                 x::GeneralizedTurnStatistic, y::GeneralizedTurnStatistic)
    GeneralizedTurnStatistic(x.p♯₋, y.p♯₊, x.ρ + y.ρ)
end

function is_turning(::TrajectoryNUTS, τ::GeneralizedTurnStatistic)
    # Uses the generalized NUTS criterion from Betancourt (2017).
    @unpack p♯₋, p♯₊, ρ = τ
    @argcheck p♯₋ ≢ p♯₊ "internal error: is_turning called on a leaf"
    dot(p♯₋, ρ) < 0 || dot(p♯₊, ρ) < 0
end

###
### leafs
###

function leaf(trajectory::TrajectoryNUTS, z, is_initial)
    @unpack H, π₀, min_Δ, turn_statistic_configuration = trajectory
    Δ = is_initial ? zero(π₀) : logdensity(H, z) - π₀
    isdiv = Δ < min_Δ
    v = leaf_acceptance_statistic(Δ, is_initial)
    if isdiv
        (z, Δ, τ), v, true
    else
        τ = leaf_turn_statistic(turn_statistic_configuration, H, z)
        (z, Δ, τ), v, false
    end
end

####
#### NUTS interface
####

"Default maximum depth for trees."
const DEFAULT_MAX_TREE_DEPTH = 10

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
                  turn_statistic_configuration = Val{:generalized}())
        @argcheck 0 < max_depth ≤ MAX_DIRECTIONS_DEPTH
        @argcheck min_Δ < 0
        S = typeof(turn_statistic_configuration)
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
struct TreeStatisticsNUTS
    "Log density (negative energy)."
    π::Float64
    "Depth of the tree."
    depth::Int
    "Reason for termination. See [`InvalidTree`](@ref) and [`REACHED_MAX_DEPTH`](@ref)."
    termination::InvalidTree
    "Acceptance rate statistic."
    acceptance_rate::Float64
    "Number of leapfrog steps evaluated."
    steps::Int
    "Directions for tree doubling (useful for debugging)."
    directions::Directions
end

"""
$(SIGNATURES)
No-U-turn Hamiltonian Monte Carlo transition, using Hamiltonian `H`, starting at evaluated
log density position `Q`, using stepsize `ϵ`. Parameters of `algorithm` govern the details
of tree construction.
Return two values, the new evaluated log density position, and tree statistics.
"""
function sample_tree(rng, algorithm::NUTS, H::Hamiltonian, Q::EvaluatedLogDensity, ϵ;
                          p = rand_p(rng, H.κ), directions = rand(rng, Directions))
    @unpack max_depth, min_Δ, turn_statistic_configuration = algorithm
    z = PhasePoint(Q, p)
    trajectory = TrajectoryNUTS(H, logdensity(H, z), ϵ, min_Δ, turn_statistic_configuration)
    ζ, v, termination, depth = sample_trajectory(rng, trajectory, z, max_depth, directions)
    tree_statistics = TreeStatisticsNUTS(logdensity(H, ζ), depth, termination,
                                         acceptance_rate(v), v.steps, directions)
    ζ.Q, tree_statistics
end

# include("NUTS.jl")


#####
##### Reporting progress.
#####

"""
$(TYPEDEF)
A placeholder type for not reporting any information.
"""
struct NoProgressReport end

"""
$(SIGNATURES)
Report to the given `reporter`.
The second argument can be
1. a string, which is displayed as is (this is supported by all reporters).
2. or a step in an MCMC chain with a known number of steps for progress reporters (see
[`make_mcmc_reporter`](@ref)).
`meta` arguments are key-value pairs.
In this context, a *step* is a NUTS transition, not a leapfrog step.
"""
report(reporter::NoProgressReport, step::Union{AbstractString,Integer}; meta...) = nothing

"""
$(SIGNATURES)
Return a reporter which can be used for progress reports with a known number of
`total_steps`. May return the same reporter, or a related object. Will display `meta` as
key-value pairs.
"""
make_mcmc_reporter(reporter::NoProgressReport, total_steps; meta...) = reporter

"""
$(TYPEDEF)
Report progress into the `Logging` framework, using `@info`.
For the information reported, a *step* is a NUTS transition, not a leapfrog step.
# Fields
$(FIELDS)
"""
Base.@kwdef struct LogProgressReport{T}
    "ID of chain. Can be an arbitrary object, eg `nothing`."
    chain_id::T = nothing
    "Always report progress past `step_interval` of the last report."
    step_interval::Int = 100
    "Always report progress past this much time (in seconds) after the last report."
    time_interval_s::Float64 = 1000.0
end

"""
$(SIGNATURES)
Assemble log message metadata.
Currently, it adds `chain_id` *iff* it is not `nothing`.
"""
_log_meta(chain_id::Nothing, meta) = meta

_log_meta(chain_id, meta) = (chain_id = chain_id, meta...)

function report(reporter::LogProgressReport, message::AbstractString; meta...)
    @info message _log_meta(reporter.chain_id, meta)...
    nothing
end

"""
$(TYPEDEF)
A composite type for tracking the state for which the last log message was emitted, for MCMC
reporting with a given total number of steps (see [`make_mcmc_reporter`](@ref).
# Fields
$(FIELDS)
"""
mutable struct LogMCMCReport{T}
    "The progress report sink."
    log_progress_report::T
    "Total steps for this stage."
    total_steps::Int
    "Index of the last reported step."
    last_reported_step::Int
    "The last time a report was logged (determined using `time_ns`)."
    last_reported_time_ns::UInt64
end

function report(reporter::LogMCMCReport, message::AbstractString; meta...)
    @info message _log_meta(reporter.log_progress_report.chain_id, meta)...
    nothing
end

function make_mcmc_reporter(reporter::LogProgressReport, total_steps::Integer; meta...)
    @info "Starting MCMC" total_steps = total_steps meta...
    LogMCMCReport(reporter, total_steps, -1, time_ns())
end

function report(reporter::LogMCMCReport, step::Integer; meta...)
    @unpack (log_progress_report, total_steps, last_reported_step,
             last_reported_time_ns) = reporter
    @unpack chain_id, step_interval, time_interval_s = log_progress_report
    @argcheck 1 ≤ step ≤ total_steps
    Δ_steps = step - last_reported_step
    t_ns = time_ns()
    Δ_time_s = (t_ns - last_reported_time_ns) / 1_000_000_000
    if last_reported_step < 0 || Δ_steps ≥ step_interval || Δ_time_s ≥ time_interval_s
        seconds_per_step = Δ_time_s / Δ_steps
        meta_progress = (step = step,
                         seconds_per_step = round(seconds_per_step; sigdigits = 2),
                         estimated_seconds_left = round((total_steps - step) *
                                                        seconds_per_step; sigdigits = 2))
        @info "MCMC progress" merge(_log_meta(chain_id, meta_progress), meta)...
        reporter.last_reported_step = step
        reporter.last_reported_time_ns = t_ns
    end
    nothing
end

"""
$(SIGNATURES)
Return a default reporter, taking the environment into account. Keyword arguments are passed
to constructors when applicable.
"""
function default_reporter(; kwargs...)
    if isinteractive()
        LogProgressReport(; kwargs...)
    else
        NoProgressReport()
    end
end

# include("reporting.jl")


#####
##### Sampling: high-level interface and building blocks
#####

"Significant digits to display for reporting."
const REPORT_SIGDIGITS = 3

####
#### parts unaffected by warmup
####

"""
$(TYPEDEF)
A log density bundled with an RNG and options for sampling. Contains the parts of the
problem which are not changed during warmup.
# Fields
$(FIELDS)
"""
struct SamplingLogDensity{P,R,L<:AbstractProbabilityModel{P},O,S}
    "Random number generator."
    rng::R
    "Log density."
    ℓ::L
    """
    Algorithm used for sampling, also contains the relevant parameters that are not affected
    by adaptation. See eg [`NUTS`](@ref).
    """
    algorithm::O
    "Reporting warmup information and chain progress."
    reporter::S
end

####
#### warmup building blocks
####

###
### warmup state
###

"""
$(TYPEDEF)
Representation of a warmup state. Not part of the API.
# Fields
$(FIELDS)
"""
struct WarmupState{P,T,L,Tκ <: KineticEnergy, Tϵ <: Union{Real,Nothing}}
    Q::EvaluatedLogDensity{P,T,L}
    κ::Tκ
    ϵ::Tϵ
end

function Base.show(io::IO, warmup_state::WarmupState)
    @unpack κ, ϵ = warmup_state
    ϵ_display = ϵ ≡ nothing ? "unspecified" : "≈ $(round(ϵ; sigdigits = REPORT_SIGDIGITS))"
    print(io, "adapted sampling parameters: stepsize (ϵ) $(ϵ_display), $(κ)")
end

###
### warmup interface and stages
###

"""
$(SIGNATURES)
Return the *results* and the *next warmup state* after warming up/adapting according to
`warmup_stage`, starting from `warmup_state`.
Use `nothing` for a no-op.
"""
function warmup(sampling_logdensity::SamplingLogDensity, warmup_stage::Nothing, warmup_state)
    nothing, warmup_state
end

"""
$(SIGNATURES)
Helper function to create random starting positions in the `[-2,2]ⁿ` box.
"""
function random_position(sptr::StackPointer, rng, ::Val{P}) where {P}
    sptr, r = PtrVector{P,Float64}(sptr)
    sptr, rand!(rng, r, -2.0, 2.0)
end

"Docstring for initial warmup arguments."
const DOC_INITIAL_WARMUP_ARGS =
"""
- `q`: initial position. *Default*: random (uniform [-2,2] for each coordinate).
- `κ`: kinetic energy specification. *Default*: Gaussian with identity matrix.
- `ϵ`: a scalar for initial stepsize, or `nothing` for heuristic finders.
"""

"""
$(SIGNATURES)
Create an initial warmup state from a random position.
# Keyword arguments
$(DOC_INITIAL_WARMUP_ARGS)
"""
function initialize_warmup_state(rng, sptr, ℓ; q = nothing, κ = nothing, ϵ = nothing)
    if q ≡ nothing
        sptr, q′ = random_position(rng, sptr, dimension(ℓ))
    else
        q′ = q
    end
    if κ ≡ nothing
        sptr, κ′ = GaussianKineticEnergy(sptr, dimension(ℓ))
    else
        κ′ = κ
    end
    sptr, eℓ = evaluate_ℓ(ℓ, q′)
    sptr, WarmupState(eℓ, κ′, ϵ)
end

"""
$(TYPEDEF)
Find a local optimum (using quasi-Newton methods).
It is recommended that this stage is applied so that the initial stepsize selection happens
in a region which is at least plausible.
"""
Base.@kwdef struct FindLocalOptimum{T}
    """
    Add `-0.5 * magnitude_penalty * sum(abs2, q)` to the log posterior **when finding the local
    optimum**. This can help avoid getting into high-density edge areas of the posterior
    which are otherwise not typical (eg multilevel models).
    """
    magnitude_penalty::T = 1e-4
    """
    Maximum number of iterations in the optimization algorithm. Recall that we don't need to
    find the mode, or even a local mode, just be in a reasonable region.
    """
    iterations::Int = 50
    # FIXME allow custom algorithm, tolerance, etc
end
@noinline ThrowOptimizationError(str) = throw(str)
function warmup!(
    sp::StackPointer, sample::AbstractMatrix, sampling_logdensity::SamplingLogDensity{D}, local_optimization::FindLocalOptimum, warmup_state
) where {D}
    @unpack ℓ, reporter = sampling_logdensity
    @unpack magnitude_penalty, iterations = local_optimization
    @unpack Q, κ, ϵ = warmup_state
    @unpack q = Q
    report(reporter, "finding initial optimum")
    sptr, ℓq = proptimize!(sp, sampling_logdensity.ℓ, q, magnitude_penalty, iterations)
    isfinite(ℓq) || ThrowOptimizationError("Optimization failed to converge, returning $ℓq.")
    # fg! = function(F, G, q)
        # ℓq, ∇ℓq = logdensity_and_gradient(ℓ, q)
        # if G ≠ nothing
            # @. G = -∇ℓq - q * magnitude_penalty
        # end
        # -ℓq - (0.5 * magnitude_penalty * sum(abs2, q))
    # end
    # objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), q)
    # opt = Optim.optimize(objective, q, Optim.LBFGS(),
                         # Optim.Options(; iterations = iterations))
    sp, q = PtrVector{D,Float64}(sp)
    ∇ℓq = PtrVector{D,Float64}(pointer(sp,Float64))
    sptr, (nothing, WarmupState(EvaluatedLogDensity(q, ℓq, ∇ℓq), κ, ϵ))
end

function warmup!(sp::StackPointer, sample::AbstractMatrix, sampling_logdensity, stepsize_search::InitialStepsizeSearch, warmup_state)
    @unpack rng, ℓ, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    # @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    z = PhasePoint(Q, rand_p(rng, κ))
    ϵ = find_initial_stepsize(stepsize_search, local_acceptance_ratio(Hamiltonian(κ, ℓ), z))
    report(reporter, "found initial stepsize",
           ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    nothing, WarmupState(Q, κ, ϵ)
end

"""
$(TYPEDEF)
Tune the step size `ϵ` during sampling, and the metric of the kinetic energy at the end of
the block. The method for the latter is determined by the type parameter `M`, which can be
1. `Diagonal` for diagonal metric (the default),
2. `Symmetric` for a dense metric,
3. `Nothing` for an unchanged metric.
# Results
A `NamedTuple` with the following fields:
- `chain`, a vector of position vectors
- `tree_statistics`, a vector of tree statistics for each sample
- `ϵs`, a vector of step sizes for each sample
# Fields
$(FIELDS)
"""
struct TuningNUTS{M,D}
    "Number of samples."
    N::Int
    "Dual averaging parameters."
    stepsize_adaptation::D
    """
    Regularization factor for normalizing variance. An estimated covariance matrix `Σ` is
    rescaled by `λ` towards ``σ²I``, where ``σ²`` is the median of the diagonal. The
    constructor has a reasonable default.
    """
    λ::Float64
    function TuningNUTS{M}(N::Integer, stepsize_adaptation::D,
                           λ = 5.0/N) where {M <: Union{Nothing,Diagonal,Symmetric},D}
        @argcheck N ≥ 20        # variance estimator is kind of meaningless for few samples
        @argcheck λ ≥ 0
        new{M,D}(N, stepsize_adaptation, λ)
    end
end

function Base.show(io::IO, tuning::TuningNUTS{M}) where {M}
    @unpack N, stepsize_adaptation, λ = tuning
    print(io, "Stepsize and metric tuner, $(N) samples, $(M) metric, regularization $(λ)")
end

Base.length(t::TuningNUTS) = t.N

"""
$(SIGNATURES)
Form a matrix from positions (`q`), with each column containing a position.
"""
position_matrix(chain) = reduce(hcat, chain)

# """
# $(SIGNATURES)
# Estimate the inverse metric from the chain.
# In most cases, this should be regularized, see [`regularize_M⁻¹`](@ref).
# """
# sample_M⁻¹(::Type{Diagonal}, chain) = Diagonal(vec(var(position_matrix(chain); dims = 2)))

# sample_M⁻¹(::Type{Symmetric}, chain) = Symmetric(cov(position_matrix(chain); dims = 2))


function warmup!(
    sp::StackPointer, sample,
    sampling_logdensity::SamplingLogDensity{D, <:VectorizedRNG.AbstractPCG},
    tuning::TuningNUTS{Diagonal},
    warmup_state
) where {D}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    L = VectorizationBase.align(D, T)
    chain = Matrix{typeof(Q.q)}(undef, L, N)
    chain_ptr = pointer(chain)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; tuning = "stepsize and Diagonal{T,PtrVector{}} metric")
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        copyto!( PtrVector{D,T}( chain_ptr ), Q.q ); chain_ptr += L*sizeof(T)
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    # κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
    sp, κ = GaussianKineticEnergy(sp, chain, λ, Val{D}())
    report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    
    ((chain = chain, tree_statistics = tree_statistics, ϵs = ϵs),
    WarmupState(Q, κ, final_ϵ(ϵ_state)))
end

function mcmc(sampling_logdensity::AbstractProbabilityModel{D}, N, warmup_state, sp = STACK_POINTER_REF[]) where {D}
    chain = Matrix{eltype(Q.q)}(undef, length(Q.q), N)
    mcmc!(chain, sampling_logdensity, N, warmup_state, sp)
end
function mcmc!(chain::AbstractMatrix, sampling_logdensity::AbstractProbabilityModel{D}, N, warmup_state, sp = STACK_POINTER_REF[]) where {D}
    @unpack rng, ℓ, sampler_options, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state

#    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    for i in 1:N
        Q, tree_statistics[i] = sample_tree(sp, rng, sampler_options, H, Q, ϵ)
        chain[:,i] .= Q.q
        
        report(mcmc_reporter, i)
    end
    (chain = chain, tree_statistics = tree_statistics)
end


# """
# $(SIGNATURES)
# Adjust the inverse metric estimated from the sample, using an *ad-hoc* shrinkage method.
# """
# function regularize_M⁻¹(Σ::Union{Diagonal,Symmetric}, λ::Real)
    # ad-hoc “shrinkage estimator”
    # (1 - λ) * Σ + λ * UniformScaling(max(1e-3, median(diag(Σ))))
# end

# function warmup(sampling_logdensity, tuning::TuningNUTS{M}, warmup_state) where {M}
    # @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    # @unpack Q, κ, ϵ = warmup_state
    # @unpack N, stepsize_adaptation, λ = tuning
    # chain = Vector{typeof(Q.q)}(undef, N)
    # tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    # H = Hamiltonian(κ, ℓ)
    # ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    # ϵs = Vector{Float64}(undef, N)
    # mcmc_reporter = make_mcmc_reporter(reporter, N; tuning = M ≡ Nothing ? "stepsize" :
                                       # "stepsize and $(M) metric")
    # for i in 1:N
        # ϵ = current_ϵ(ϵ_state)
        # ϵs[i] = ϵ
        # Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        # chain[i] = Q.q
        # tree_statistics[i] = stats
        # ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        # report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    # end
    # if M ≢ Nothing
        # κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
        # report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    # end
    # ((chain = chain, tree_statistics = tree_statistics, ϵs = ϵs),
     # WarmupState(Q, κ, final_ϵ(ϵ_state)))
# end

# """
# $(SIGNATURES)
# Markov Chain Monte Carlo for `sampling_logdensity`, with the adapted `warmup_state`.
# Return a `NamedTuple` of
# - `chain`, a vector of length `N` that contains the positions,
# - `tree_statistics`, a vector of length `N` with the tree statistics.
# """
# function mcmc(sampling_logdensity, N, warmup_state)
    # @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    # @unpack Q, κ, ϵ = warmup_state
    # chain = Vector{typeof(Q.q)}(undef, N)
    # tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    # H = Hamiltonian(κ, ℓ)
    # mcmc_reporter = make_mcmc_reporter(reporter, N)
    # for i in 1:N
        # Q, tree_statistics[i] = sample_tree(rng, algorithm, H, Q, ϵ)
        # chain[i] = Q.q
        # report(mcmc_reporter, i)
    # end
    # (chain = chain, tree_statistics = tree_statistics)
# end

function mcmc(sampling_logdensity::SamplingLogDensity{D}, N, warmup_state, sp = STACK_POINTER_REF[]) where {D}
    chain = Matrix{eltype(Q.q)}(undef, length(Q.q), N)
    mcmc!(chain, sampling_logdensity, N, warmup_state, sp)
end
function mcmc!(chain::AbstractMatrix, sampling_logdensity::SamplingLogDensity{D}, N, warmup_state, sp = STACK_POINTER_REF[]) where {D}
    @unpack rng, ℓ, sampler_options, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state

#    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    for i in 1:N
        Q, tree_statistics[i] = sample_tree(sp, rng, sampler_options, H, Q, ϵ)
        chain[:,i] .= Q.q
        
        report(mcmc_reporter, i)
    end
    (chain = chain, tree_statistics = tree_statistics)
end



"""
$(SIGNATURES)
Helper function for constructing the “middle” doubling warmup stages in
[`default_warmup_stages`](@ref).
"""
function _doubling_warmup_stages(M, stepsize_adaptation, middle_steps,
                                 doubling_stages::Val{D}) where {D}
    ntuple(i -> TuningNUTS{M}(middle_steps * 2^(i - 1), stepsize_adaptation), Val(D))
end

"""
$(SIGNATURES)
A sequence of warmup stages:
1. find the local optimum using `local_optimization`,
2. select an initial stepsize using `stepsize_search` (default: based on a heuristic),
3. tuning stepsize with `init_steps` steps
4. tuning stepsize and covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times
5. tuning stepsize with `terminating_steps` steps.
`M` (`Diagonal`, the default or `Symmetric`) determines the type of the metric adapted from
the sample.
This is the suggested tuner of most applications.
Use `nothing` for `local_optimization` or `stepsize_adaptation` to skip the corresponding
step.
"""
function default_warmup_stages(;
                               local_optimization = FindLocalOptimum(),
                               stepsize_search = InitialStepsizeSearch(),
                               M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                               stepsize_adaptation = DualAveraging(),
                               init_steps = 75, middle_steps = 25, doubling_stages::Val{DS} = Val(5),
                               terminating_steps = 50) where {DS}
    (local_optimization, stepsize_search,
     TuningNUTS{Nothing}(init_steps, stepsize_adaptation),
     _doubling_warmup_stages(M, stepsize_adaptation, middle_steps, Val(DS))...,
     TuningNUTS{Nothing}(terminating_steps, stepsize_adaptation))
end

"""
$(SIGNATURES)
A sequence of warmup stages for fixed stepsize:
1. find the local optimum using `local_optimization`,
2. tuning covariance: first with `middle_steps` steps, then repeat with twice
   the steps `doubling_stages` times
Very similar to [`default_warmup_stages`](@ref), but omits the warmup stages with just
stepsize tuning.
"""
function fixed_stepsize_warmup_stages(;
                                      local_optimization = FindLocalOptimum(),
                                      M::Type{<:Union{Diagonal,Symmetric}} = Diagonal,
                                      middle_steps = 25, doubling_stages::Val{DS} = 5) where {DS}
    (local_optimization,
     _doubling_warmup_stages(M, FixedStepsize(), middle_steps, Val(DS))...)
end

"""
$(SIGNATURES)
Helper function for implementing warmup.
!!! note
    Changes may imply documentation updates in [`mcmc_keep_warmup`](@ref).
"""
function _warmup!(sptr, sample, sampling_logdensity, stages, initial_warmup_state)
    foldl(stages; init = ((), initial_warmup_state)) do acc, stage
        stages_and_results, warmup_state = acc
        results, warmup_state′ = warmup!(sptr, sample, sampling_logdensity, stage, warmup_state)
        stage_information = (stage = stage, results = results, warmup_state = warmup_state′)
        (stages_and_results..., stage_information), warmup_state′
    end
end

"Shared docstring part for the MCMC API."
const DOC_MCMC_ARGS =
"""
# Arguments
- `rng`: the random number generator, eg `Random.GLOBAL_RNG`.
- `ℓ`: the log density, supporting the API of the `LogDensityProblems` package
- `N`: the number of samples for inference, after the warmup.
# Keyword arguments
- `initialization`: see below.
- `warmup_stages`: a sequence of warmup stages. See [`default_warmup_stages`](@ref) and
  [`fixed_stepsize_warmup_stages`](@ref); the latter requires an `ϵ` in initialization.
- `algorithm`: see [`NUTS`](@ref). It is very unlikely you need to modify
  this, except perhaps for the maximum depth.
- `reporter`: how progress is reported. By default, verbosely for interactive sessions using
  the log message mechanism (see [`LogProgressReport`](@ref), and no reporting for
  non-interactive sessions (see [`NoProgressReport`](@ref)).
# Initialization
The `initialization` keyword argument should be a `NamedTuple` which can contain the
following fields (all of them optional and provided with reasonable defaults):
$(DOC_INITIAL_WARMUP_ARGS)
"""

"""
$(SIGNATURES)
Perform MCMC with NUTS, keeping the warmup results. Returns a `NamedTuple` of
- `initial_warmup_state`, which contains the initial warmup state
- `warmup`, an iterable of `NamedTuple`s each containing fields
    - `stage`: the relevant warmup stage
    - `results`: results returned by that warmup stage (may be `nothing` if not applicable,
      or a chain, with tree statistics, etc; see the documentation of stages)
    - `warmup_state`: the warmup state *after* the corresponding stage.
- `final_warmup_state`, which contains the final adaptation after all the warmup
- `inference`, which has `chain` and `tree_statistics`, see [`mcmc_with_warmup`](@ref).
!!! warning
    This function is not (yet) exported because the the warmup interface may change with
    minor versions without being considered breaking. Recommended for interactive use.
$(DOC_MCMC_ARGS)
"""
function mcmc_keep_warmup(rng::AbstractRNG, sptr::StackPointer, ℓ, N::Integer;
                          initialization = (),
                          warmup_stages = default_warmup_stages(),
                          algorithm = NUTS(),
                          reporter = default_reporter())
    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    sptr, initial_warmup_state = initialize_warmup_state(rng, sptr, ℓ; initialization...)
    warmup, warmup_state = _warmup(sampling_logdensity, warmup_stages, initial_warmup_state)
    inference = mcmc(sampling_logdensity, N, warmup_state)
    (initial_warmup_state = initial_warmup_state, warmup = warmup,
     final_warmup_state = warmup_state, inference = inference)
end

"""
$(SIGNATURES)
Perform MCMC with NUTS, including warmup which is not returned. Return a `NamedTuple` of
- `chain`, a vector of positions from the posterior
- `tree_statistics`, a vector of tree statistics
- `κ` and `ϵ`, the adapted metric and stepsize.
$(DOC_MCMC_ARGS)
# Usage examples
Using a fixed stepsize:
```julia
mcmc_with_warmup(rng, ℓ, N;
                 initialization = (ϵ = 0.1, ),
                 warmup_stages = fixed_stepsize_warmup_stages())
```
Starting from a given position `q₀` and kinetic energy scaled down (will still be adapted):
```julia
mcmc_with_warmup(rng, ℓ, N;
                 initialization = (q = q₀, κ = GaussianKineticEnergy(5, 0.1)))
```
Using a dense metric:
```julia
mcmc_with_warmup(rng, ℓ, N;
                 warmup_stages = default_warmup_stages(; M = Symmetric))
```
Disabling the optimization step:
```julia
mcmc_with_warmup(rng, ℓ, N;
                 warmup_stages = default_warmup_stages(; local_optimization = nothing,
                                                         M = Symmetric))
```
"""
function mcmc_with_warmup(rng, ℓ, N; initialization = (),
                          warmup_stages = default_warmup_stages(),
                          algorithm = NUTS(), reporter = default_reporter())
    @unpack final_warmup_state, inference =
        mcmc_keep_warmup(rng, ℓ, N; initialization = initialization,
                         warmup_stages = warmup_stages, algorithm = algorithm,
                         reporter = reporter)
    @unpack κ, ϵ = final_warmup_state
    (inference..., κ = κ, ϵ = ϵ)
end

function mcmc_with_warmup!(
    rng::AbstractRNG, sptr::StackPointer, sample::AbstractMatrix, ℓ::AbstractProbabilityModel, N = size(sample,2);
    initialization = (), warmup_stages = default_warmup_stages(), algorithm = NUTS(), reporter = default_reporter()
)

    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    sptr, initial_warmup_state = initialize_warmup_state(rng, sptr, ℓ; initialization...)
    warmup, warmup_state = _warmup!(sptr, sample, sampling_logdensity, warmup_stages, initial_warmup_state)
    inference = mcmc(sampling_logdensity, N, warmup_state)
    (initial_warmup_state = initial_warmup_state, warmup = warmup,
     final_warmup_state = warmup_state, inference = inference)


    @unpack final_warmup_state, inference =
        mcmc_keep_warmup(
            rng, ℓ, N; initialization = initialization,
            warmup_stages = warmup_stages, algorithm = algorithm,
            reporter = reporter
        )
    @unpack κ, ϵ = final_warmup_state
    (inference..., κ = κ, ϵ = ϵ)
end


function threaded_mcmc(
    ℓ::AbstractProbabilityModel{D}, N; initialization = (),
    warmup_stages = default_warmup_stages(),
    algorithm = NUTS(), reporter = default_reporter()
) where {D}
    nthreads = ProbabilityModels.NTHREADS[]
    sptr = ProbabilityModels.STACK_POINTER_REF[]
    LSS = ProbabilityModels.LOCAL_STACK_SIZE[]
    nwarmup = maximum(length, warmup_stages)
    NS = max(N,nwarmup)
    L = VectorizationBase.align(D, Float64)
    samples = DynamicPaddedArray{Float64}(undef, (D, NS, nthreads), L)
    sample_ptr = pointer(samples)
    Threads.@threads for t in 1:nthreads
        sample = DynamicPtrMatrix{Float64}(sample_ptr, (D, NS), L)
        rng =  ProbabilityModels.GLOBAL_PCGs[t]
        mcmc_with_warmup!(rng, sptr, sample, ℓ, N; initialization = initialization, warmup_stages = warmup_stages, algorithm = algorithm, reporter = reporter)
        sptr += LSS
        sample_ptr += sizeof(Float64)*NS*L
    end
    samples
    @unpack final_warmup_state, inference = mcmc_keep_warmup(
        rng, ℓ, N; initialization = initialization, warmup_stages = warmup_stages, algorithm = algorithm, reporter = reporter)
    @unpack κ, ϵ = final_warmup_state
    (inference..., κ = κ, ϵ = ϵ)
end


# include("mcmc.jl")

#####
##### statistics and diagnostics
#####

module Diagnostics

export EBFMI, summarize_tree_statistics, explore_log_acceptance_ratios, leapfrog_trajectory,
    InvalidTree, REACHED_MAX_DEPTH, is_divergent

using DynamicHMC: GaussianKineticEnergy, Hamiltonian, evaluate_ℓ, InvalidTree,
    REACHED_MAX_DEPTH, is_divergent, log_acceptance_ratio, PhasePoint, rand_p, leapfrog,
    logdensity, MAX_DIRECTIONS_DEPTH

using ArgCheck: @argcheck
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using LogDensityProblems: dimension
using Parameters: @unpack
import Random
using Statistics: mean, quantile, var

"""
$(SIGNATURES)
Energy Bayesian fraction of missing information. Useful for diagnosing poorly
chosen kinetic energies.
Low values (`≤ 0.3`) are considered problematic. See Betancourt (2016).
"""
function EBFMI(tree_statistics)
    πs = map(x -> x.π, tree_statistics)
    mean(abs2, diff(πs)) / var(πs)
end

"Acceptance quantiles for [`TreeStatisticsSummary`](@ref) diagnostic summary."
const ACCEPTANCE_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

"""
$(TYPEDEF)
Storing the output of [`NUTS_statistics`](@ref) in a structured way, for pretty
printing. Currently for internal use.
# Fields
$(FIELDS)
"""
struct TreeStatisticsSummary{T <: Real, C <: NamedTuple}
    "Sample length."
    N::Int
    "average_acceptance"
    a_mean::T
    "acceptance quantiles"
    a_quantiles::Vector{T}
    "termination counts"
    termination_counts::C
    "depth counts (first element is for `0`)"
    depth_counts::Vector{Int}
end

"""
$(SIGNATURES)
Count termination reasons in `tree_statistics`.
"""
function count_terminations(tree_statistics)
    max_depth = 0
    divergence = 0
    turning = 0
    for tree_statistic in tree_statistics
        it = tree_statistic.termination
        if it == REACHED_MAX_DEPTH
            max_depth += 1
        elseif is_divergent(it)
            divergence += 1
        else
            turning += 1
        end
    end
    (max_depth = max_depth, divergence = divergence, turning = turning)
end

"""
$(SIGNATURES)
Count depths in tree statistics.
"""
function count_depths(tree_statistics)
    c = zeros(Int, MAX_DIRECTIONS_DEPTH + 1)
    for tree_statistic in tree_statistics
        c[tree_statistic.depth + 1] += 1
    end
    c[1:something(findlast(!iszero, c), 0)]
end

"""
$(SIGNATURES)
Summarize tree statistics. Mostly useful for NUTS diagnostics.
"""
function summarize_tree_statistics(tree_statistics)
    As = map(x -> x.acceptance_rate, tree_statistics)
    TreeStatisticsSummary(length(tree_statistics),
                          mean(As), quantile(As, ACCEPTANCE_QUANTILES),
                          count_terminations(tree_statistics),
                          count_depths(tree_statistics))
end

function Base.show(io::IO, stats::TreeStatisticsSummary)
    @unpack N, a_mean, a_quantiles, termination_counts, depth_counts = stats
    println(io, "Hamiltonian Monte Carlo sample of length $(N)")
    print(io, "  acceptance rate mean: $(round(a_mean; digits = 2)), 5/25/50/75/95%:")
    for aq in a_quantiles
        print(io, " ", round(aq; digits = 2))
    end
    println(io)
    function print_percentages(pairs)
        is_first = true
        for (key, value) in sort(collect(pairs), by = first)
            if is_first
                is_first = false
            else
                print(io, ",")
            end
            print(io, " $(key) => $(round(Int, 100*value/N))%")
        end
    end
    print(io, "  termination:")
    print_percentages(pairs(termination_counts))
    println(io)
    print(io, "  depth:")
    print_percentages(zip(axes(depth_counts, 1) .- 1, depth_counts))
end

####
#### Acceptance ratio diagnostics
####

"""
$(SIGNATURES)
From an initial position, calculate the uncapped log acceptance ratio for the given log2
stepsizes and momentums `ps`, `N` of which are generated randomly by default.
"""
function explore_log_acceptance_ratios(ℓ, q, log2ϵs;
                                       rng = Random.GLOBAL_RNG,
                                       κ = GaussianKineticEnergy(dimension(ℓ)),
                                       N = 20, ps = [rand_p(rng, κ) for _ in 1:N])
    H = Hamiltonian(κ, ℓ)
    Q = evaluate_ℓ(ℓ, q)
    [log_acceptance_ratio(H, PhasePoint(Q, p), 2.0^log2ϵ) for log2ϵ in log2ϵs, p in ps]
end

"""
$(TYPEDEF)
Implements an iterator on a leapfrog trajectory until the first non-finite log density.
# Fields
$(FIELDS)
"""
struct LeapfrogTrajectory{TH,TZ,TF,Tϵ}
    "Hamiltonian"
    H::TH
    "Initial position"
    z₀::TZ
    "Negative energy at initial position."
    π₀::TF
    "Stepsize (negative: move backward)."
    ϵ::Tϵ
end

Base.IteratorSize(::Type{<:LeapfrogTrajectory}) = Base.SizeUnknown()

function Base.iterate(lft::LeapfrogTrajectory, zi = (lft.z₀, 0))
    @unpack H, ϵ, π₀ = lft
    z, i = zi
    if isfinite(z.Q.ℓq)
        z′ = leapfrog(H, z, ϵ)
        i′ = i + sign(ϵ)
        _position_information(lft, z′, i′), (z′, i′)
    else
        nothing
    end
end

"""
$(SIGNATURES)
Position information returned by [`leapfrog_trajectory`](@ref), see documentation there.
Internal function.
"""
function _position_information(lft::LeapfrogTrajectory, z, i)
    @unpack H, π₀ = lft
    (z = z, position = i, Δ = logdensity(H, z) - π₀)
end

"""
$(SIGNATURES)
Calculate a leapfrog trajectory visiting `positions` (specified as a `UnitRange`, eg `-5:5`)
relative to the starting point `q`, with stepsize `ϵ`. `positions` has to contain `0`, and
the trajectories are only tracked up to the first non-finite log density in each direction.
Returns a vector of `NamedTuple`s, each containin
- `z`, a [`PhasePoint`](@ref) object,
- `position`, the corresponding position,
- `Δ`, the log density + the kinetic energy relative to position `0`.
"""
function leapfrog_trajectory(ℓ, q, ϵ, positions::UnitRange{<:Integer};
                             rng = Random.GLOBAL_RNG,
                             κ = GaussianKineticEnergy(dimension(ℓ)), p = rand_p(rng, κ))
    A, B = first(positions), last(positions)
    @argcheck A ≤ 0 ≤ B "Positions has to contain `0`."
    Q = evaluate_ℓ(ℓ, q)
    H = Hamiltonian(κ, ℓ)
    z₀ = PhasePoint(Q, p)
    π₀ = logdensity(H, z₀)
    lft_fwd = LeapfrogTrajectory(H, z₀, π₀, ϵ)
    fwd_part = collect(Iterators.take(lft_fwd, B))
    bwd_part = collect(Iterators.take(LeapfrogTrajectory(H, z₀, π₀, -ϵ), -A))
    vcat(reverse!(bwd_part), _position_information(lft_fwd, z₀, 0), fwd_part)
end

end

# include("diagnostics.jl")





end # module
