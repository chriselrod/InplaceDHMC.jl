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
        # @vvectorize instead of @simd ivdep for the masked reduction
        @vvectorize_unsafe $T 4 for d ∈ 1:$D # these are PtrVector already; GC preservation must happen elsewhere
            pᵈ = p[d]
            ke += pᵈ * M⁻¹[d] * pᵈ
        end
        $T(0.5) * ke
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
    @inbounds @simd ivdep for l ∈ 1:L
        M⁻¹p[l] = M⁻¹[l] * p[l]
    end
    sptr, M⁻¹p
end
function calculate_p♯(tree::Tree{P,T,L}, κ::GaussianKineticEnergy, p::PtrVector{P,T,L}, q = nothing) where {P,T,L}
    M⁻¹ = κ.M⁻¹.diag
    M⁻¹p = undefined_ρ♯( tree )
    @inbounds @simd ivdep for l ∈ 1:L
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
rand_p!(rng::VectorizedRNG.AbstractPCG, r::PaddedMatrices.AbstractMutableFixedSizeVector{P,T,L}, κ::GaussianKineticEnergy{P,T,L}, q = nothing) where {P,T,L} = randn!(rng, r, κ.W.diag)
rand_p(rng::AbstractRNG, κ::GaussianKineticEnergy{P,T,L}, q = nothing) where {P,T,L} = randn!(rng, FixedSizeVector{P,T,L}(undef), κ)


"""
$(SIGNATURES)
Evaluate log density and gradient and save with the position. Preferred interface for
creating `EvaluatedLogDensity` instances.
"""
function evaluate_ℓ!(sptr::StackPointer, ∇ℓq::PtrVector{P,T,L,false}, ℓ::AbstractProbabilityModel{P}, q::PtrVector{P,T,L,false}) where {P,T,L}
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
    @fastmath @inbounds @simd ivdep for l ∈ 1:L
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
    @fastmath @inbounds @simd ivdep for l ∈ 1:L
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
    @fastmath @inbounds @simd ivdep for l ∈ 1:L
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
    @fastmath @inbounds @simd ivdep for l ∈ 1:L
        pₘ[l] = pₘ[l] + ϵₕ * ∇ℓq′[l]
    end
    sp, PhasePoint(Q′, pₘ)
end

