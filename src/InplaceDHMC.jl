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
using Random
using Statistics: cov, mean, median, middle, quantile, var

using VectorizationBase, LoopVectorization, VectorizedRNG, StackPointers, PaddedMatrices, QuasiNewtonMethods, ProbabilityModels, SIMDPirates
using QuasiNewtonMethods: AbstractProbabilityModel, dimension, logdensity, logdensity_and_gradient!
using PaddedMatrices: Static
# using ProbabilityModels: 
# using PaddedMatrices: AbstractFixedSizePaddedMatrix

# copy from StatsFuns.jl
function logaddexp(x, y)
    isfinite(x) && isfinite(y) || return x > y ? x : y # QuasiNewtonMethods.nanmax(x,y)
    x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
end


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
    M⁻¹::Diagonal{T,PtrVector{P,T,L,true}}
    "W such that W*W'=M. Used for generating random draws."
    W::Diagonal{T,PtrVector{P,T,L,true}}
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
function GaussianKineticEnergy(sptr::StackPointer, M⁻¹::Diagonal{T,PtrVector{P,T,L,true}}) where {P,T,L}
    sptr, W = PtrVector{P,T,L}(sptr)
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
function GaussianKineticEnergy(sptr::StackPointer, ::Static{D}, m⁻¹::T = 1.0) where {D,T}
    sptr, M⁻¹ = PtrVector{D,T}(sptr)
    sptr, W   = PtrVector{D,T}(sptr)
    @fastmath mroot = 1 / sqrt(m⁻¹)
    @inbounds for d in 1:PaddedMatrices.full_length(W)
        M⁻¹[d] = m⁻¹
        W[d] = mroot
    end
    # @show M⁻¹, W
    # @show pointer(M⁻¹), pointer(W)
    sptr, GaussianKineticEnergy(Diagonal(M⁻¹), Diagonal(W))
end


function regularized_cov_block_quote(W::Int, T, reps_per_block::Int, stride::Int, mask_last::Bool = false, mask::Unsigned = 0xff)
    # loads from ptr_sample
    # stores in ptr_s² and ptr_invs
    # needs vNinv, mulreg, and addreg to be defined
    reps_per_block -= 1
    size_T = sizeof(T)
    WT = size_T*W
    V = Vec{W,T}
    quote
        $([Expr(:(=), Symbol(:μ_,i), :(SIMDPirates.vload($V, ptr_sample + $(WT*i), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...)))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:Σδ_,i), :(SIMDPirates.vbroadcast($V,zero($T)))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:Σδ²_,i), :(SIMDPirates.vbroadcast($V,zero($T)))) for i ∈ 0:reps_per_block]...)
        for n ∈ 1:N-1
            $([Expr(:(=), Symbol(:δ_,i), :(SIMDPirates.vsub(SIMDPirates.vload($V, ptr_sample + $(WT*i) + n*$stride*$size_T),$(Symbol(:μ_,i))))) for i ∈ 0:reps_per_block]...)
            $([Expr(:(=), Symbol(:Σδ_,i), :(SIMDPirates.vadd($(Symbol(:δ_,i)),$(Symbol(:Σδ_,i))))) for i ∈ 0:reps_per_block]...)
            $([Expr(:(=), Symbol(:Σδ²_,i), :(SIMDPirates.vmuladd($(Symbol(:δ_,i)),$(Symbol(:δ_,i)),$(Symbol(:Σδ²_,i))))) for i ∈ 0:reps_per_block]...)
        end
        $([Expr(:(=), Symbol(:ΣδΣδ_,i), :(SIMDPirates.vmul($(Symbol(:Σδ_,i)),$(Symbol(:Σδ_,i))))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:s²nm1_,i), :(SIMDPirates.vfnmadd($(Symbol(:ΣδΣδ_,i)),vNinv,$(Symbol(:Σδ²_,i))))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:regs²_,i), :(SIMDPirates.vmuladd($(Symbol(:s²nm1_,i)), vmulreg, vaddreg))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:reginvs_,i), :(SIMDPirates.rsqrt($(Symbol(:regs²_,i))))) for i ∈ 0:reps_per_block]...)
        $([:(vstore!(ptr_s² + $(WT*i), $(Symbol(:regs²_,i)), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...))) for i ∈ 0:reps_per_block]...)
        $([:(vstore!(ptr_invs + $(WT*i), $(Symbol(:reginvs_,i)), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...))) for i ∈ 0:reps_per_block]...)
    end
end

@generated function GaussianKineticEnergy(sp::StackPointer, sample::AbstractMatrix{T}, λ::T, ::Val{D}) where {D,T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W-1
    M = (D + Wm1) & ~Wm1
    AL = VectorizationBase.align(M*size_T)
    quote
        ptr_s² = pointer(sp, $T)
        regs² = PtrVector{$M,$T}(ptr_s²)
        ptr_invs = ptr_s² + $AL
        invs = PtrVector{$M,$T}(ptr_invs)
        sp + $(2AL), GaussianKineticEnergy!(regs², invs, sample, λ)
    end
end

GaussianKineticEnergy!(K::GaussianKineticEnergy, sample::AbstractMatrix, λ) = GaussianKineticEnergy!(K.M⁻¹.diag, K.W.diag, sample, λ)

@generated function GaussianKineticEnergy!(
    regs²::PtrVector{D,T,L}, invs::PtrVector{D,T,L}, sample::AbstractMatrix{T}, λ::T
# ) where {D,L,T}
) where {D,T,L}
    W, Wshift = VectorizationBase.pick_vector_width_shift(L, T)
    Wm1 = W-1
    # M = (D + Wm1) & ~Wm1
    M = L
    V = Vec{W,T}
    MdW = (M + Wm1) >> Wshift
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
    # AL = VectorizationBase.align(M*size_T)
    q = quote
        # ptr_s² = pointer(sp, $T)
        # regs² = PtrVector{$M,$T}(ptr_s²)
        # ptr_invs = ptr_s² + $AL
        # invs = PtrVector{$M,$T}(ptr_invs)
        ptr_s² = pointer(regs²)
        ptr_invs = pointer(invs)
        # @show size(sample)
        # display(sample)
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
    if blocked_reps > 0
        loop_block = regularized_cov_block_quote(W, T, reps_per_block, M, false)
        body_quote = quote
            $loop_block
            ptr_sample += $WT*$reps_per_block
            ptr_s² += $WT*$reps_per_block
            ptr_invs += $WT*$reps_per_block
        end
        if blocked_reps > 1
            body_quote = quote
                for _ ∈ 1:$blocked_reps
                    $body_quote
                end
            end
        end
        push!(q.args, body_quote)
    end
    if blocked_rem > 0
        push!(q.args, regularized_cov_block_quote(W, T, blocked_rem, M, need_to_mask, VectorizationBase.mask(T,Wrem)))        
    end
    # push!(q.args, :(@show regs²))
    # push!(q.args, :(@show invs))
    push!(q.args, :(GaussianKineticEnergy(Diagonal(regs²), Diagonal(invs))))
    # display(q)
    q
end

function Base.show(io::IO, κ::GaussianKineticEnergy{D,T}) where {D,T}
    print(io::IO, "Gaussian kinetic energy ($D,$T), √diag(M⁻¹): $(.√(diag(κ.M⁻¹)))")
end

## NOTE about implementation: the 3 methods are callable without a third argument (`q`)
## because they are defined for Gaussian (Euclidean) kinetic energies.

Base.size(κ::GaussianKineticEnergy{P}) where {P} = (P,P)
# Base.size(κ::GaussianKineticEnergy, args...) = size(κ.M⁻¹, args...)


####
#### Hamiltonian
####

struct Hamiltonian{P,T,N,L}
    "The kinetic energy specification."
    κ::GaussianKineticEnergy{P,T,N}
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
    function Hamiltonian(κ::GaussianKineticEnergy{P,T,N}, ℓ::L) where {P,T,N,L<:AbstractProbabilityModel{P}}
        # @argcheck capabilities(ℓ) ≥ LogDensityOrder(1)
        # @argcheck dimension(ℓ) == size(κ, 1)
        new{P,T,N,L}(κ, ℓ)
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
    q::PtrVector{P,T,L,true}
    "ℓ(q). Saved for reuse in sampling."
    ℓq::T
    "∇ℓ(q). Cached for reuse in sampling."
    ∇ℓq::PtrVector{P,T,L,true}
    function EvaluatedLogDensity(q::PtrVector{P,T,L,true}, ℓq::T, ∇ℓq::PtrVector{P,T,L,true}) where {P,T<:Real,L}
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
$(TYPEDEF)
A point in phase space, consists of a position (in the form of an evaluated log density `ℓ`
at `q`) and a momentum.
"""
struct PhasePoint{D,T,L}
    "Evaluated log density."
    Q::EvaluatedLogDensity{D,T,L}
    "Momentum."
    p::PtrVector{D,T,L,true}
    flag::UInt32
    # function PhasePoint(Q::EvaluatedLogDensity, p::S) where {T,S}
        # @argcheck length(p) == length(Q.q)
        # new{T,S}(Q, p)
    # end
end
PhasePoint(Q, p) = PhasePoint(Q, p, 0x00000000)

"Default maximum depth for trees."
const DEFAULT_MAX_TREE_DEPTH = 10

"""
The Tree data structure manages memory while sampling. In particular, preallocation is tricky while sampling from trees. Exploring a tree of depth N
will require evaluating 2^N positions, and calculating gradients and momentums for each, as well as turning statistics.
The tree allocates enough space for the maximum possible number of live variables (depth + 1), which is much smaller than 2^N for N > 2, and keeps
track of which are still live or have been freed and can thus be reused.

Supports a treedepth of up to 30. Code can be modified to support up to 62, but if you're anywhere near 30 you've probably got severe sampling issues.
The default maximum depth is 10. Runtime is exponential as a function of the depth you're hitting; a depth of 30 will take about a million times longer than 10.

The limit comes from using 32 bits to indicate whether each vector is allocated or not, allowing space for depth+2 vectors.
A 1 bit indicates the corresponding space is available; 0 means occupied.
"""
struct Tree{D,T,L}
    root::Ptr{T}
    depth::Int
    sptr::StackPointer
end
aligned_offset(::Tree{D,T,L}) where {D,T,L} = L*sizeof(T)
# struct FlaggedPhasePoint{D,T,L}
    # z::PhasePoint{D,T,L}
    # flag::UInt32
# end
struct FlaggedVector{D,T,L} <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{D,T,L}
    v::PtrVector{D,T,L,true}
    flag::UInt32
end
@inline function FlaggedVector{D,T,L}(ptr::Ptr{T}, flag::UInt32) where {D,T,L}
    FlaggedVector{D,T,L}(PtrVector{D,T,L,true}(ptr), flag)
end
@inline Base.pointer(v::FlaggedVector) = v.v.ptr

function Tree{D,T,L}(sptr::StackPointer, depth::Int = DEFAULT_MAX_TREE_DEPTH) where {D,T,L}
    root = pointer(sptr, T)
    depth₊ = depth + 3
    # set roots to zero so that all loads can properly be interpreted as bools without further processing on future accesses.
    SIMDPirates.vstore!(reinterpret(Ptr{UInt32}, root), (VE(0xffffffff),VE(0xffffffff),VE(0xffffffff),VE(0xffffffff)))
    Tree{D,T,L}( root, depth₊, sptr + VectorizationBase.REGISTER_SIZE + 6L*sizeof(T)*depth₊ )
end
clear!(tree::Tree) = SIMDPirates.vstore!(reinterpret(Ptr{UInt64}, tree.root), (VE(0xffffffffffffffff),VE(0xffffffffffffffff)))
clear_all_but_z!(tree::Tree, flag::UInt32) = SIMDPirates.vstore!(reinterpret(Ptr{UInt32}, tree.root), (VE(0xffffffff ⊻ flag),VE(0xffffffff),VE(0xffffffff),VE(0xffffffff)))

@generated Tree{D,T}(sptr::StackPointer, depth::Int = DEFAULT_MAX_TREE_DEPTH) where {D,T} = :(Tree{$D,$T,$(VectorizationBase.align(D,T))}(sptr, depth))
Tree(sptr::StackPointer, depth::Int, ::PtrVector{D,T,L}) where {D,T,L} = Tree{D,T,L}(sptr, dept)

"""
To allocate, we find the first unoccupied slot (corresponding to a bit of 1), and then set that bit to 0.
"""
function allocate!(root::Ptr)
    root32 = reinterpret(Ptr{UInt32}, root)
    allocations = VectorizationBase.load(root32)
    first_unallocated = leading_zeros(allocations)
    flag = 0x80000000 >> first_unallocated
    VectorizationBase.store!(root32, allocations ⊻ flag)
    first_unallocated, flag
end
function allocate!(tree::Tree, flag::UInt32, i=0)
    root32 = reinterpret(Ptr{UInt32}, tree.root) + i
    allocations = VectorizationBase.load(root32)
    VectorizationBase.store!(root32, allocations ⊻ flag)
end
function isallocated(tree::Tree, flag::UInt32, i=0)
    allocations = VectorizationBase.load(reinterpret(Ptr{UInt32}, tree.root))
    (allocations | flag) != allocations
end
function undefined_z(tree::Tree{D,T,L}) where {D,T,L}
    @unpack root = tree
    LT = aligned_offset(tree)
    stump = root + VectorizationBase.REGISTER_SIZE
    first_unallocated, flag = allocate!(root)
    # @show first_unallocated, flag
    # @assert first_unallocated < tree.depth
    # display(flag)
    # print("Defining z at $first_unallocated with flag "); display(flag)
    # println("Defining z at $first_unallocated with flag $(bitstring(flag))")
    # first_unallocated < tree.depth || println("Warning, results invalid: Defining z at $first_unallocated")
    # first_unallocated < tree.depth || display(stacktrace())
    @assert first_unallocated < tree.depth "Don't have space to allocate z!"
    stump + 3first_unallocated*LT, flag
end
function undefined_ρ♯(tree::Tree{D,T,L}) where {D,T,L}
    @unpack root, depth = tree
    LT = aligned_offset(tree)
    stump = root + VectorizationBase.REGISTER_SIZE + 4LT*depth
    first_unallocated, flag = allocate!(root + 8)
    # @assert first_unallocated < depth
    # first_unallocated < tree.depth || println("Warning, results invalid: Defining ρ♯ at $first_unallocated")
    # println("Defining ρ♯ at $first_unallocated with flag $(bitstring(flag))")
    # first_unallocated < tree.depth || display(stacktrace())
    @assert first_unallocated < (depth<<1) "Don't have space to allocate ρ♯!"
    FlaggedVector{D,T,L}( stump + first_unallocated*LT, flag )
end
function undefined_Σρ(tree::Tree{D,T,L}) where {D,T,L}
    @unpack root, depth = tree
    LT = aligned_offset(tree)
    stump = root + VectorizationBase.REGISTER_SIZE + 3LT*depth
    first_unallocated, flag = allocate!(root + 4)
    # @assert first_unallocated < depth
    # first_unallocated < tree.depth || println("Warning, results invalid: Defining Σρ at $first_unallocated")
    # first_unallocated < tree.depth || display(stacktrace())
    @assert first_unallocated < depth "Don't have space to allocate Σρ!"
    FlaggedVector{D,T,L}( stump + first_unallocated*LT, flag )
end
"""
To free, we simply set the bit back to 1 so that it may be allocated again.
"""
function free!(tree::Tree, flag::UInt32, offset::Int)
    @unpack root = tree
    # println("Freeing flag $(bitstring(flag)) at offset $offset")
    # print("Freeing flag "); display(flag); println(" at offset $offset")
    root32 = reinterpret(Ptr{UInt32}, root) + offset
    allocated = VectorizationBase.load(root32)
    flag != 0x00000000 && @assert (allocated | flag) != allocated
    VectorizationBase.store!(root32, VectorizationBase.load(root32) | flag)
    nothing
end
free_z!(tree::Tree, flag::UInt32) = free!(tree, flag, 0)
free_ρ♯!(tree::Tree, flag::UInt32) = free!(tree, flag, 8)
free_Σρ!(tree::Tree, flag::UInt32) = free!(tree, flag, 4)
# free_z!(tree::Tree, flag::UInt32) = (print("free z called on flag: "); display(flag); free!(tree, flag, 0))
# function free_z!(tree::Tree, flag::UInt32)

# end
# free_ρ♯!(tree::Tree, flag::UInt32) = (println("free ρ♯ called"); free!(tree, flag, 4))
# free_Σρ!(tree::Tree, flag::UInt32) = (println("free Σρ called"); free!(tree, flag, 8))

# @inline Base.pointer(z::PhasePoint) = pointer(z.p)
# @inline StackPointers.StackPointer(z::PhasePoint{P,T,L}) where {P,T,L} = StackPointer(z.Q.∇ℓq.ptr + L*sizeof(T))

#####
##### Abstract tree/trajectory interface
#####

####
#### Directions
####

"Maximum number of iterations [`next_direction`](@ref) supports."
const MAX_DIRECTIONS_DEPTH = 32

"""
Internal type implementing random directions. Draw a new value with `rand`, see
[`next_direction`](@ref).
Serves two purposes: a fixed value of `Directions` is useful for unit testing, and drawing a
single bit flag collection economizes on the RNG cost.
"""
struct Directions
    flags::UInt32
end

Base.rand(rng::AbstractRNG, ::Type{Directions}) = Directions(rand(rng, UInt32))
Base.rand(rng::VectorizedRNG.AbstractPCG, ::Type{Directions}) = Directions(rand(rng, UInt32))

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
function combine_turn_statistics_in_direction(tree::Tree, trajectory, τ₁, τ₂, is_forward::Bool)
    if is_forward
        combine_turn_statistics(tree, trajectory, τ₁, τ₂)
    else
        combine_turn_statistics(tree, trajectory, τ₂, τ₁)
    end
end

function combine_proposals_and_logweights(
    rng, tree::Tree, trajectory, ζ₁, ζ₂, ω₁::Real, ω₂::Real, is_forward::Bool, is_doubling::Bool
)
    ω = logaddexp(ω₁, ω₂)
    logprob2 = calculate_logprob2(trajectory, is_doubling, ω₁, ω₂, ω)
    ζ = combine_proposals(rng, tree, trajectory, ζ₁, ζ₂, logprob2, is_forward, is_doubling)
    ζ, ω
end
# function combine_proposals_and_logweights(
    # rng, tree::Tree, trajectory, ζ₁, ζ₂, ω₁::Real, ω₂::Real, is_forward::Bool
# )
    # ω = logaddexp(ω₁, ω₂)
    # logprob2 = calculate_logprob2(trajectory, false, ω₁, ω₂, ω)
    # ζ = combine_proposals(rng, tree, trajectory, ζ₁, ζ₂, logprob2, is_forward)
    # ζ, ω
# end

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
    left::Int32
    right::Int32
end

InvalidTree(i::Integer) = InvalidTree(Base.unsafe_trunc(Int32, i), Base.unsafe_trunc(Int32, i))

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
const REACHED_MAX_DEPTH = InvalidTree(one(Int32), zero(Int32))

# on tree initialization, store


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
function adjacent_tree(rng, tree::Tree{P,T,L}, trajectory, z::PhasePoint{P,T,L}, i::Int32, depth::Int32, is_forward::Bool) where {P,T,L}
    i′ = i + (is_forward ? one(Int32) : -one(Int32) )
    # @show (1, bitstring(unsafe_load(reinterpret(Ptr{UInt32}, tree.root), 2)))
    # @show z.Q.q
    # @show depth, i′
    lb, ub = 5, 10
    # lb <= abs(i′) < ub && @show z
    if depth == zero(Int32) # moves from z into ζready
        z′ = move(tree, trajectory, z, is_forward)
        # lb <= abs(i′) < ub && @show logdensity(trajectory.H, z′), trajectory.π₀
        (ζ, ω, τ), v, invalid = leaf(tree, trajectory, z′, false)
        return (ζ, ω, τ, z′, i′), v, (invalid,InvalidTree(i′))
    else
        # “left” tree
        t₋, v₋, (invalid,it) = adjacent_tree(rng, tree, trajectory, z, i, depth - one(Int32), is_forward)
        # @show first(t₋)
        # @show t₋[4]
        invalid && return t₋, v₋, (invalid, it)
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        # “right” tree — visited information from left is kept even if invalid
        t₊, v₊, (invalid,it) = adjacent_tree(rng, tree, trajectory, z₋, i₋, depth - one(Int32), is_forward)
        v = combine_visited_statistics(trajectory, v₋, v₊)
        invalid && return t₊, v, (invalid,it)
        ζ₊, ω₊, τ₊, z₊, i₊ = t₊

        # turning invalidates
        # try
        # @show (2, bitstring(unsafe_load(reinterpret(Ptr{UInt32}, tree.root), 2)))
        τ = combine_turn_statistics_in_direction(tree, trajectory, τ₋, τ₊, is_forward)
        # catch err
        #     @show 
        #     rethrow(err)
        is_turning(trajectory, τ) && return t₊, v, (true, InvalidTree(i′, i₊))

        # valid subtree, combine proposals
        ζ, ω = combine_proposals_and_logweights(rng, tree, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false)
        return (ζ, ω, τ, z₊, i₊), v, (false,REACHED_MAX_DEPTH)
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
function sample_trajectory(rng, tree::Tree, trajectory, zᵢ::PhasePoint{P,T,L}, max_depth::Integer, directions::Directions) where {P,T,L}
    #    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    # original_flag = zᵢ.flag
    # protect_initial = true # Protect initial position by giving it a dummy flag
    # z = PhasePoint(zᵢ.Q, zᵢ.p, 0x00000000)
    z = zᵢ
    # @show logdensity(trajectory.H, z), trajectory.π₀
    (ζ, ω, τ), v, invalid = leaf(tree, trajectory, z, true)
    z₋ = z₊ = z
    # z₋flag = z₊flag = 0x00000000
    depth = zero(Int32)
    termination = REACHED_MAX_DEPTH
    i₋ = i₊ = zero(Int32)
    # dealloc₋ = dealloc₊ = false
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        # @show depth, is_forward
        if is_forward
            zᵢ, iᵢ = z₊, i₊
            alloc = z₋.flag
        else
            zᵢ, iᵢ = z₋, i₋
            alloc = z₊.flag
        end
        # ((ζ.flag === zᵢ) | (z₊.flag === z₋.flag)) || free_z!(tree, zᵢ.flag)
        VectorizationBase.store!(reinterpret(Ptr{UInt32}, tree.root), 0xffffffff ⊻ (alloc | ζ.flag))
        # clear_all_but_z!( tree, alloc | ζ.flag )
        t′, v′, (invalid, it) = adjacent_tree(
            rng, tree, trajectory, zᵢ, iᵢ, depth, is_forward
        )
        
        v = combine_visited_statistics(trajectory, v, v′)

        # invalid adjacent tree: stop
        invalid && (termination = it; break)

        # extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′
        guard′ = z′.flag === ζ′.flag
        allocate!(tree, z′.flag)
        # ζ′g = z′.flag === ζ′.flag ? PhasePoint(ζ′.Q, ζ′.p, 0x00000000) : ζ′
        # update edges and combine proposals
        if is_forward
            z₊, i₊ = z′, i′
        else
            z₋, i₋ = z′, i′
        end

        # tree has doubled successfully
        ζ, ω = combine_proposals_and_logweights(
            rng, tree, trajectory, ζ, ζ′, ω, ω′, is_forward, true
        )
        depth += one(Int32)

        # when the combined tree is turning, stop
        τ = combine_turn_statistics_in_direction(tree, trajectory, τ, τ′, is_forward)
        is_turning(trajectory, τ) && (termination = InvalidTree(i₋, i₊); break)
    end
    # @show  original_flag, ζ.flag
    # clear_all_but_z!( tree, ζ.flag )# == 0x00000000 ? original_flag : ζ.flag )
    clear!( tree )
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
$(SIGNATURES)
Return kinetic energy `κ`, at momentum `p`.
"""
@generated function kinetic_energy(κ::GaussianKineticEnergy{D,T,L}, p::PtrVector{D,T,L}, q = nothing) where {D,T,L}
    quote
        M⁻¹ = κ.M⁻¹.diag
        # @show M⁻¹
        # @show p
        ke = zero(T)
        # @vvectorize instead of @simd for the masked reduction
        @vvectorize $T 4 for d ∈ 1:$D
            pᵈ = p[d]
            ke += pᵈ * M⁻¹[d] * pᵈ
        end
        T(0.5) * ke
    end
end
# kinetic_energy(κ::GaussianKineticEnergy, p, q = nothing) = dot(p, κ.M⁻¹ * p) / 2

"""
$(SIGNATURES)
Return ``p♯ = M⁻¹⋅p``, used for turn diagnostics.
"""
function calculate_p♯(sptr::StackPointer, κ::GaussianKineticEnergy, p::PtrVector{P,T,L}, q = nothing) where {P,T,L}
    M⁻¹ = κ.M⁻¹
    sptr, M⁻¹p = PtrVector{P,T,L}(sptr)
    @inbounds @simd for l ∈ 1:L
        M⁻¹p[l] = M⁻¹[l] * p[l]
    end
    sptr, M⁻¹p
end
function calculate_p♯(tree::Tree{P,T,L}, κ::GaussianKineticEnergy, p::PtrVector{P,T,L}, q = nothing) where {P,T,L}
    M⁻¹ = κ.M⁻¹.diag
    M⁻¹p = undefined_ρ♯( tree )
    @inbounds @simd for l ∈ 1:L
        M⁻¹p[l] = M⁻¹[l] * p[l]
    end
    M⁻¹p
end

# """
# $(SIGNATURES)
# Calculate the gradient of the logarithm of kinetic energy in momentum `p`.
# """
# ∇kinetic_energy(sptr::StackPointer, κ::GaussianKineticEnergy, p, q = nothing) = calculate_p♯(sptr, κ, p)

"""
$(SIGNATURES)
Generate a random momentum from a kinetic energy at position `q`.
"""
function rand_p(rng::AbstractRNG, sptr::StackPointer, κ::GaussianKineticEnergy{P,T,L}, q = nothing) where {P,T,L}
    W = κ.W
    sptr, r = PtrVector{P,T,L}(sptr)
    sptr, randn!(rng, r, W)
end
rand_p!(rng::VectorizedRNG.AbstractPCG, r::PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T,L}, κ::GaussianKineticEnergy{P,T,L}, q = nothing) where {P,T,L} = randn!(rng, r, κ.W.diag)
rand_p(rng::AbstractRNG, κ::GaussianKineticEnergy{P,T,L}, q = nothing) where {P,T,L} = randn!(rng, MutableFixedSizePaddedVector{P,T,L}(undef), κ)


"""
$(SIGNATURES)
Evaluate log density and gradient and save with the position. Preferred interface for
creating `EvaluatedLogDensity` instances.
"""
function evaluate_ℓ!(sptr::StackPointer, ∇ℓq::PtrVector{P,T,L,true}, ℓ::AbstractProbabilityModel{P}, q::PtrVector{P,T,L,true}) where {P,T,L}
    ℓq = logdensity_and_gradient!(∇ℓq, ℓ, q, sptr)
    sp = reinterpret(Int, pointer(sptr, Float64))
    dlq = reinterpret(Int, pointer(∇ℓq))
    dq = reinterpret(Int, pointer(q))
    # @show sp
    # @show (sp - dlq) >> 3
    # @show (sp - dq) >> 3
    if isfinite(ℓq)
        EvaluatedLogDensity(q, ℓq, ∇ℓq)
    else
        EvaluatedLogDensity(q, oftype(ℓq, -Inf), q) # second q used just as a placeholder
    end
end

function evaluate_ℓ(sptr::StackPointer, ℓ::AbstractProbabilityModel{P}, q::PtrVector{P,T,L}) where {P,T,L}
    sptr2, ∇ℓq = PtrVector{P,T,L}(sptr)
    ℓq = logdensity_and_gradient!(∇ℓq, ℓ, q, sptr2)
    if isfinite(ℓq)
        sptr2, EvaluatedLogDensity(q, ℓq, ∇ℓq)
    else
        sptr, EvaluatedLogDensity(q, oftype(ℓq, -Inf), q) # second q used just as a placeholder
    end
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
function QuasiNewtonMethods.logdensity(H::Hamiltonian{D,T,L}, z::PhasePoint{D,T,L}) where {D,T,L}
    @unpack ℓq = z.Q
    isfinite(ℓq) || return oftype(ℓq, -Inf)
    K = kinetic_energy(H.κ, z.p)
    ℓq - (isfinite(K) ? K : oftype(K, Inf))
end

function calculate_p♯(stack, H::Hamiltonian{D,T,L}, z::PhasePoint{D,T,L}) where {D,T,L}
    calculate_p♯(stack, H.κ, z.p)
end

"""
    leapfrog(H, z, ϵ)
Take a leapfrog step of length `ϵ` from `z` along the Hamiltonian `H`.
Return the new phase point.
The leapfrog algorithm uses the gradient of the next position to evolve the momentum. If
this is not finite, the momentum won't be either, `logdensity` above will catch this and
return an `-Inf`, making the point divergent.
"""
function leapfrog(tree::Tree{P,T,L},
        H::Hamiltonian{P,T,L},
        z::PhasePoint{P,T,L}, ϵ::T
) where {P,L,T}
    @unpack ℓ, κ = H
    @unpack p, Q = z
    @unpack q, ∇ℓq = Q
#    @argcheck isfinite(Q.ℓq) "Internal error: leapfrog called from non-finite log density"
    # @show bitstring(z.flag)
    treeptr, flag = undefined_z(tree)
    # @show bitstring(flag)
    LT = L*sizeof(T) # counting on this being aligned.
    pₘ = PtrVector{P,T,L}(treeptr) 
    q′ = PtrVector{P,T,L}(treeptr + LT)
    # @show ϵ
    # @show z
    # @show bitstring(z.flag)
    M⁻¹ = κ.M⁻¹.diag
    ϵₕ = T(0.5) * ϵ
    @fastmath @inbounds @simd for l ∈ 1:L
        pₘₗ = p[l] + ϵₕ * ∇ℓq[l]
        pₘ[l] = pₘₗ
        q′[l] = q[l] + ϵ * M⁻¹[l] * pₘₗ
    end
    # @show q′
    # Variables that escape:
    # p′, Q′ (q′, ∇ℓq)
    Q′ = evaluate_ℓ!(tree.sptr, PtrVector{P,T,L}(treeptr + 2LT), H.ℓ, q′) # ∇ℓq is sorted second
    # @show Q′.q
    # isfinite(Q′.ℓq) || return PhasePoint(Q′, pₘ, flag)
    # p′ = pₘ # PtrVector{P,T,L}(sptr + 3LT)
    ∇ℓq′ = Q′.∇ℓq
    @fastmath @inbounds @simd for l ∈ 1:L
        pₘ[l] = pₘ[l] + ϵₕ * ∇ℓq′[l]
    end
    PhasePoint(Q′, pₘ, flag)
end
function leapfrog(sp::StackPointer,
        H::Hamiltonian{P,T,L},
        z::PhasePoint{P,T,L}, ϵ::T
) where {P,L,T}
    @unpack ℓ, κ = H
    @unpack p, Q = z
    @unpack q, ∇ℓq = Q
#    @argcheck isfinite(Q.ℓq) "Internal error: leapfrog called from non-finite log density"
    sptr = pointer(sp, T)
    LT = L*sizeof(T) # counting on this being aligned.
    pₘ = PtrVector{P,T,L}(sptr) 
    q′ = PtrVector{P,T,L}(sptr + LT)
    M⁻¹ = κ.M⁻¹.diag
    ϵₕ = T(0.5) * ϵ
    @fastmath @inbounds @simd for l ∈ 1:L
        pₘₗ = p[l] + ϵₕ * ∇ℓq[l]
        pₘ[l] = pₘₗ
        q′[l] = q[l] + ϵ * M⁻¹[l] * pₘₗ
    end
    # Variables that escape:
    # p′, Q′ (q′, ∇ℓq)
    sp, Q′ = evaluate_ℓ(sp + 2LT, H.ℓ, q′) # ∇ℓq is sorted second
    # isfinite(Q′.ℓq) || return PhasePoint(Q′, pₘ)
    ∇ℓq′ = Q′.∇ℓq
    # p′ = pₘ # PtrVector{P,T,L}(sptr + 3LT)
    @fastmath @inbounds @simd for l ∈ 1:L
        pₘ[l] = pₘ[l] + ϵₕ * ∇ℓq′[l]
    end
    sp, PhasePoint(Q′, pₘ)
end


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
        # @argcheck 0 < a_min < a_max < 1
        # @argcheck 0 < ϵ₀
        # @argcheck 1 < C
        # @argcheck maxiter_crossing ≥ 50
        # @argcheck maxiter_bisect ≥ 50
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
    # @show s, a, C
    for _ in 1:maxiter_crossing
        ϵ = ϵ₀ * C
        Aϵ = A(ϵ)
        # @show Aϵ, ϵ
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
    # @argcheck ϵ₀ < ϵ₁
    # @argcheck Aϵ₀ > a_max && Aϵ₁ < a_min
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
    logdensity(H, leapfrog(sptr, H, z, ϵ)) - target
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
function local_acceptance_ratio(sptr, H, z)
    target = logdensity(H, z)
    isfinite(target) ||
        ThrowDomainError(z.p, "Starting point has non-finite density.")
    ϵ -> exp(logdensity(H, last(leapfrog(sptr, H, z, ϵ))) - target)
    # ϵ -> begin
        # ld = logdensity(H, last(leapfrog(sptr, H, z, ϵ)))
        # @show ld, target
        # exp(ld - target)
    # end
end

function find_initial_stepsize(sptr::StackPointer, parameters::InitialStepsizeSearch, H, z)
    find_initial_stepsize(parameters, local_acceptance_ratio(sptr, H, z))
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
        # @argcheck 0 < δ < 1
        # @argcheck γ > 0
        # @argcheck 0.5 < κ ≤ 1
        # @argcheck t₀ ≥ 0
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
    # @argcheck ϵ > 0
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
    # @argcheck 0 ≤ a ≤ 1
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

function move(tree::Tree{D,T,L}, trajectory::TrajectoryNUTS, z::PhasePoint{D,T,L}, fwd::Bool) where {D,T,L}
    @unpack H, ϵ = trajectory
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

function combine_proposals(rng, tree::Tree, ::TrajectoryNUTS, z₁, z₂, logprob2::Real, is_forward::Bool, is_doubling::Bool)
    z, flag = rand_bool_logprob(rng, logprob2) ? (z₂, z₁.flag) : (z₁, z₂.flag)
    is_doubling || free_z!(tree, flag)
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
    @inbounds @simd for l in 1:L
        ρ[l] = ρₓ[l] + ρʸ[l]
    end
    # x.p♯₊.flag == x.p♯₋.flag || free_ρ♯!(tree, x.p♯₊.flag)
    # y.p♯₊.flag == y.p♯₋.flag || free_ρ♯!(tree, y.p♯₋.flag)
    GeneralizedTurnStatistic(x.p♯₋, y.p♯₊, ρ)
end


@generated function is_turning(::TrajectoryNUTS, τ::GeneralizedTurnStatistic{D,T,L}) where {D,T,L}
    quote
    # Uses the generalized NUTS criterion from Betancourt (2017).
        @unpack p♯₋, p♯₊, ρ = τ
    # @argcheck p♯₋ ≢ p♯₊ "internal error: is_turning called on a leaf"
        d♯₋ = zero($T)
        d♯₊ = zero($T)
        @vvectorize $T 2 for d in 1:$D
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
        (d♯₋ < zero($T)) | (d♯₊ < zero($T))
    end
end

###
### leafs
###

function leaf(tree::Tree, trajectory::TrajectoryNUTS, z, is_initial)
    @unpack H, π₀, min_Δ, turn_statistic_configuration = trajectory
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
    trajectory = TrajectoryNUTS(H, logdensity(H, z), ϵ, min_Δ, turn_statistic_configuration)
    ζ, v, termination, depth = sample_trajectory(rng, tree, trajectory, z, max_depth, directions)
    tree_statistics = TreeStatisticsNUTS(logdensity(H, ζ), acceptance_rate(v), termination, depth, v.steps)#, directions)
    ζ, tree_statistics
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
@inline report(reporter::NoProgressReport, step::Union{AbstractString,Integer}; meta...) = nothing

"""
$(SIGNATURES)
Return a reporter which can be used for progress reports with a known number of
`total_steps`. May return the same reporter, or a related object. Will display `meta` as
key-value pairs.
"""
@inline make_mcmc_reporter(reporter::NoProgressReport, total_steps; meta...) = reporter

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
    # @argchecky 1 ≤ step ≤ total_steps
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
struct SamplingLogDensity{D,R,L<:AbstractProbabilityModel{D},O,S}
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
struct WarmupState{D,T,L,Tκ <: KineticEnergy, Tϵ <: Union{Real,Nothing}}
    z::PhasePoint{D,T,L}
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
function warmup!(tree, chain, sampling_logdensity::SamplingLogDensity, warmup_stage::Nothing, warmup_state)
    nothing, warmup_state
end

random_position!(rng::AbstractRNG, q::PtrVector) = rand!(rng, q, -2.0, 2.0)

"""
$(SIGNATURES)
Helper function to create random starting positions in the `[-2,2]ⁿ` box.
"""
function random_position(rng::AbstractRNG, sptr::StackPointer, ::Static{P}) where {P}
    sptr, q = PtrVector{P,Float64}(sptr)
    sptr, random_position!(rng, q)
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
function initialize_warmup_state(rng, sptr::StackPointer, ℓ::AbstractProbabilityModel{D}; q = nothing, κ = nothing, ϵ = nothing) where {D}
    if κ ≡ nothing
        sptr, κ′ = GaussianKineticEnergy(sptr, dimension(ℓ))
    else
        κ′ = κ
    end
    # @show κ′.W.diag, κ′.M⁻¹.diag
    # @show pointer(κ′.W.diag), pointer(κ′.M⁻¹.diag)
    tree = Tree{D,Float64}(sptr)
    # println("Allocated tree:")
    # @show κ′.W.diag, κ′.M⁻¹.diag
    # @show pointer(κ′.W.diag), pointer(κ′.M⁻¹.diag)
    treeptr, flag = undefined_z(tree)
    # @show treeptr
    LT = aligned_offset(tree)
    p = PtrVector{D,Float64}(treeptr)
    # println("About to generate a random position.")
    # @show κ′.W.diag, κ′.M⁻¹.diag
    # @show pointer(κ′.W.diag), pointer(κ′.M⁻¹.diag)    
    q′ = q ≡ nothing ? random_position!(rng, PtrVector{D,Float64}(treeptr + LT)) : q
    ∇ℓq = PtrVector{D,Float64}(treeptr + 2LT)
    # println("About to evaluate the logdensity:")
    # @show κ′.W.diag, κ′.M⁻¹.diag
    # @show pointer(κ′.W.diag), pointer(κ′.M⁻¹.diag)
    Q = evaluate_ℓ!(tree.sptr, ∇ℓq, ℓ, q′)
    # println("Just evaluated the logdensity:")
    # @show κ′.W.diag, κ′.M⁻¹.diag
    # @show pointer(κ′.W.diag), pointer(κ′.M⁻¹.diag)
    tree, WarmupState(PhasePoint(Q, p, flag), κ′, ϵ)
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
    tree::Tree, chain, sampling_logdensity::SamplingLogDensity{D}, local_optimization::FindLocalOptimum, warmup_state
) where {D}
    @unpack ℓ, reporter = sampling_logdensity
    @unpack magnitude_penalty, iterations = local_optimization
    @unpack z, κ, ϵ = warmup_state
    @unpack Q, p, flag = z
    @unpack q, ℓq, ∇ℓq = Q
    report(reporter, "finding initial optimum")
    ℓq = QuasiNewtonMethods.proptimize!(tree.sptr, ℓ, q, ∇ℓq, ℓq, magnitude_penalty, iterations)#+100)
    # @show q
    # @show ℓq
    # @show ∇ℓq
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
    # sp, q = PtrVector{D,Float64}(sp)
    # ∇ℓq = PtrVector{D,Float64}(pointer(sp,Float64))
    # sptr is set ahead by proptimize! to store optim and gradient.
    nothing, WarmupState(PhasePoint(EvaluatedLogDensity(q, ℓq, ∇ℓq), p, flag), κ, ϵ)
end
Base.length(::FindLocalOptimum) = 0
function warmup!(tree::Tree, chain, sampling_logdensity, stepsize_search::InitialStepsizeSearch, warmup_state)
    @unpack rng, ℓ, reporter = sampling_logdensity
    @unpack z, κ, ϵ = warmup_state
    # @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    rand_p!(rng, z.p, κ)
    ϵ = find_initial_stepsize(stepsize_search, local_acceptance_ratio(tree.sptr, Hamiltonian(κ, ℓ), z))
    report(reporter, "found initial stepsize",
           ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    nothing, WarmupState(z, κ, ϵ)
end
Base.length(::InitialStepsizeSearch) = 0
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
        # @argcheck N ≥ 20        # variance estimator is kind of meaningless for few samples
        # @argcheck λ ≥ 0
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

# function store_sample!(_∇ℓq::PtrVector{P,T,L}, _q::PtrVector{P,T,L}, Q::EvaluatedLogDensity) where {P,T,L}
    # @unpack q, ∇ℓq = Q
    # @inbounds for l in 1:L
        # _q[l] = q[l]
        # _∇ℓq[l] = ∇ℓq[l]
    # end
    # EvaluatedLogDensity(
        # _q, Q.ℓq, _∇ℓq
    # )
# end

function warmup!(
    tree::Tree{D,T,L}, chain₊::AbstractMatrix{T},
    sampling_logdensity::SamplingLogDensity{D},
    tuning::TuningNUTS{M},
    warmup_state
) where {D,T,L,M}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack z, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    # L = VectorizationBase.align(D, T)
    chain_ptr = pointer(chain₊)
    chain = DynamicPtrMatrix{T}(chain_ptr, (D, N), L)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; tuning = "stepsize and Diagonal{T,PtrVector{}} metric")
    # sp, ∇ℓq = PtrVector{D,T}(sp)
    for n in 1:N
        ϵ = current_ϵ(ϵ_state)
        # @assert ϵ > 1e-10 "Current ϵ: $ϵ; final: $(final_ϵ(ϵ_state))"
        if ϵ < 1e-10
            @show z
            Q = evaluate_ℓ!(tree.sptr, z.Q.∇ℓq, sampling_logdensity.ℓ, z.Q.q)
            @show Q
            throw(AssertionError("Current ϵ: $ϵ; final: $(final_ϵ(ϵ_state))"))
        end
        ϵs[n] = ϵ
        z, stats = sample_tree(rng, tree, algorithm, H, z, ϵ)
        copyto!( PtrVector{D,T}( chain_ptr ), z.Q.q ) # relocate to base of stack
        # @show z.Q.q
        chain_ptr += L*sizeof(T)
        tree_statistics[n] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, n; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    # κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
    # sp, κ = GaussianKineticEnergy(sp, chain, λ, Val{D}())
    if M ≢ Nothing
        GaussianKineticEnergy!(κ, chain, λ)
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    end
    (chain = chain, tree_statistics = tree_statistics, ϵs = ϵs), WarmupState(z, κ, final_ϵ(ϵ_state))
end

# function mcmc(sampling_logdensity::AbstractProbabilityModel{D}, N, warmup_state, sp = STACK_POINTER_REF[]) where {D}
    # chain = Matrix{eltype(Q.q)}(undef, length(Q.q), N)
    # mcmc!(chain, sampling_logdensity, N, warmup_state, sp)
# end
function mcmc!(tree::Tree{D,T,L}, chain::AbstractMatrix, sampling_logdensity::SamplingLogDensity{D}, N, warmup_state) where {D,T,L}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack z, κ, ϵ = warmup_state

    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    chain_ptr = pointer(chain)
    for n in 1:N
        z, tree_statistics[n] = sample_tree(rng, tree, algorithm, H, z, ϵ)
        copyto!( PtrVector{D,T}( chain_ptr ), z.Q.q )
        chain_ptr += L*sizeof(T)
        
        report(mcmc_reporter, n)
    end
    tree_statistics
end



"""
$(SIGNATURES)
Helper function for constructing the “middle” doubling warmup stages in
[`default_warmup_stages`](@ref).
"""
function _doubling_warmup_stages(M, stepsize_adaptation, middle_steps,
                                 doubling_stages::Val{D}) where {D}
    ntuple(d -> TuningNUTS{M}(middle_steps << (d - 1), stepsize_adaptation), Val(D))
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
@generated function _warmup!(tree::Tree, chain, sampling_logdensity, stages::T, initial_warmup_state) where {T}
    N = length(T.parameters)
    q = quote
        acc_0 = (), initial_warmup_state
    end
    for n in 1:N
        warmup_call_q = quote
            ($(Symbol(:stages_and_results_,n-1)), $(Symbol(:warmup_state_,n-1))) = $(Symbol(:acc_,n-1))
            $(Symbol(:stage_,n)) = stages[$n]
            ($(Symbol(:results_,n)), $(Symbol(:warmup_state′_,n))) = warmup!(tree, chain, sampling_logdensity, $(Symbol(:stage_,n)), $(Symbol(:warmup_state_,n-1)))
            $(Symbol(:stage_information_,n)) = (stage = $(Symbol(:stage_,n)), results = $(Symbol(:results_,n)), warmup_state = $(Symbol(:warmup_state′_,n)))
            $(Symbol(:acc_,n)) = ($(Symbol(:stages_and_results_,n-1))..., $(Symbol(:stage_information_,n))), $(Symbol(:warmup_state′_,n))
        end
        push!(q.args, warmup_call_q)
    end
    push!(q.args, Symbol(:acc_,N))
    q
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
function mcmc_keep_warmup(rng::AbstractRNG, tree::Tree, ℓ, N::Integer;
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
    rng::AbstractRNG, sptr::StackPointer, chain::AbstractMatrix{T}, ℓ::AbstractProbabilityModel{D}, N = size(chain,2);
    initialization = (), warmup_stages = default_warmup_stages(), algorithm = NUTS(), reporter = default_reporter()
) where {D,T}

    # We allocate the tree here.
    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    tree, initial_warmup_state = initialize_warmup_state(rng, sptr, ℓ; initialization...)
    warmup, warmup_state = _warmup!(tree, chain, sampling_logdensity, warmup_stages, initial_warmup_state)
    tree_statistics = mcmc!(tree, chain, sampling_logdensity, N, warmup_state)
    # (initial_warmup_state = initial_warmup_state, warmup = warmup,
     # final_warmup_state = warmup_state, inference = inference)


    # @unpack final_warmup_state, inference =
        # mcmc_keep_warmup(
            # rng, ℓ, N; initialization = initialization,
            # warmup_stages = warmup_stages, algorithm = algorithm,
            # reporter = reporter
        # )
    # @unpack κ, ϵ = final_warmup_state
    # (inference..., κ = κ, ϵ = ϵ)
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
    chains = DynamicPaddedArray{Float64}(undef, (D, NS, nthreads), L)
    chain_ptr = pointer(chains)
    # Threads.@threads
    for t in 0:nthreads-1
        chain = DynamicPtrMatrix{Float64}(chain_ptr + t*8NS*L, (D, NS), L)
        rng =  ProbabilityModels.GLOBAL_PCGs[t+1]
        mcmc_with_warmup!(rng, sptr + t*LSS, chain, ℓ, N; initialization = initialization, warmup_stages = warmup_stages, algorithm = algorithm, reporter = reporter)
    end
    chains
    # @unpack final_warmup_state, inference = mcmc_keep_warmup(
        # rng, ℓ, N; initialization = initialization, warmup_stages = warmup_stages, algorithm = algorithm, reporter = reporter)
    # @unpack κ, ϵ = final_warmup_state
    # (inference..., κ = κ, ϵ = ϵ)
end


# include("mcmc.jl")


end # module
