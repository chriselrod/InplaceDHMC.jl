
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
    M⁻¹::Diagonal{T,PtrVector{P,T,L,false}}
    "W such that W*W'=M. Used for generating random draws."
    W::Diagonal{T,PtrVector{P,T,L,false}}
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
function GaussianKineticEnergy(sptr::StackPointer, M⁻¹::Diagonal{T,PtrVector{P,T,L,false}}) where {P,T,L}
    sptr, W = PtrVector{P,T,L}(sptr)
    M⁻¹d = M⁻¹.diag
    @fastmath @inbounds @simd ivdep for l ∈ 1:L
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
    q::PtrVector{P,T,L,false}
    "ℓ(q). Saved for reuse in sampling."
    ℓq::T
    "∇ℓ(q). Cached for reuse in sampling."
    ∇ℓq::PtrVector{P,T,L,false}
    function EvaluatedLogDensity(q::PtrVector{P,T,L,false}, ℓq::T, ∇ℓq::PtrVector{P,T,L,false}) where {P,T<:Real,L}
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
    p::PtrVector{D,T,L,false}
    flag::UInt32
    # function PhasePoint(Q::EvaluatedLogDensity, p::S) where {T,S}
        # @argcheck length(p) == length(Q.q)
        # new{T,S}(Q, p)
    # end
end
PhasePoint(Q, p) = PhasePoint(Q, p, 0x00000000)

