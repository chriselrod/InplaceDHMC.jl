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
    tree::Tree, chain, tree_statistics, ϵs, sampling_logdensity::SamplingLogDensity{D},
    local_optimization::FindLocalOptimum, warmup_state
) where {D}
    @unpack rng, ℓ, reporter = sampling_logdensity
    @unpack magnitude_penalty, iterations = local_optimization
    @unpack z, κ, ϵ = warmup_state
    @unpack Q, p, flag = z
    @unpack q, ℓq, ∇ℓq = Q
    report(reporter, "finding initial optimum")
    for _ in 1:100
        ℓq = QuasiNewtonMethods.proptimize!(tree.sptr, ℓ, q, ∇ℓq, ℓq, magnitude_penalty, iterations)#+100)
        # @show q
        # @show ℓq
        # @show ∇ℓq
        isfinite(ℓq) && return WarmupState(PhasePoint(EvaluatedLogDensity(q, ℓq, ∇ℓq), p, flag), κ, ϵ)
        random_position!(rng, q)
        ℓq = evaluate_ℓ!(tree.sptr, ∇ℓq, ℓ, q).ℓq
    end
    ThrowOptimizationError("Optimization failed to converge, returning $ℓq; Thrad ID: $(Threads.threadid()).")
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
end
Base.length(::FindLocalOptimum) = 0
function warmup!(
    tree::Tree, chain, tree_statistics, ϵs,
    sampling_logdensity, stepsize_search::InitialStepsizeSearch, warmup_state
)
    @unpack rng, ℓ, reporter = sampling_logdensity
    @unpack z, κ, ϵ = warmup_state
    # @argcheck ϵ ≡ nothing "stepsize ϵ manually specified, won't perform initial search"
    rand_p!(rng, z.p, κ)
    ϵ = find_initial_stepsize(stepsize_search, local_acceptance_ratio(tree.sptr, Hamiltonian(κ, ℓ), z))
    report(reporter, "found initial stepsize",
           ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    WarmupState(z, κ, ϵ)
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

# """
# $(SIGNATURES)
# Form a matrix from positions (`q`), with each column containing a position.
# """
# position_matrix(chain) = reduce(hcat, chain)

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
    tree_statistics::AbstractVector{TreeStatisticsNUTS}, ϵs::AbstractVector{T},
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
    # tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    # ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; tuning = "stepsize and Diagonal{T,PtrVector{}} metric")
    # sp, ∇ℓq = PtrVector{D,T}(sp)
    for n in 1:N
        ϵ = current_ϵ(ϵ_state)
        # @assert ϵ > 1e-10 "Current ϵ: $ϵ; final: $(final_ϵ(ϵ_state))"
        if ϵ < 1e-10
            @show Threads.threadid(), z
            Q = evaluate_ℓ!(tree.sptr, z.Q.∇ℓq, sampling_logdensity.ℓ, z.Q.q)
            @show Threads.threadid(), Q
            throw(AssertionError("Current ϵ: $ϵ; final: $(final_ϵ(ϵ_state))"))
        end
        ϵs[n] = ϵ
        z, stats = sample_tree(rng, tree, algorithm, H, z, ϵ)
        copyto!( PtrVector{D,T}( chain_ptr ), z.Q.q ) # relocate to base of stack
        # @show z.Q.q
        chain_ptr += L*sizeof(T)
        tree_statistics[n] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        reporting(mcmc_reporter) && report(mcmc_reporter, n; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    # κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
    # sp, κ = GaussianKineticEnergy(sp, chain, λ, Val{D}())
    if M ≢ Nothing
        GaussianKineticEnergy!(κ, chain, λ)
        reporting(mcmc_reporter) && report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    end
    # (chain = chain, tree_statistics = tree_statistics, ϵs = ϵs)
    WarmupState(z, κ, final_ϵ(ϵ_state))
end

function mcmc!(tree::Tree{D,T,L}, chain::AbstractMatrix, tree_statistics::AbstractVector{TreeStatisticsNUTS}, sampling_logdensity::SamplingLogDensity{D}, N, warmup_state) where {D,T,L}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack z, κ, ϵ = warmup_state

    # tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    chain_ptr = pointer(chain)
    for n in 1:N
        z, tree_statistics[n] = sample_tree(rng, tree, algorithm, H, z, ϵ)
        copyto!( PtrVector{D,T}( chain_ptr ), z.Q.q )
        chain_ptr += L*sizeof(T)
        
        report(mcmc_reporter, n)
    end
    # tree_statistics
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
@generated function _warmup!(tree::Tree, chain, tree_statistics, ϵs, sampling_logdensity, stages::T, warmup_state_0) where {T}
    N = length(T.parameters)
    q = quote end
    for n in 1:N
        warmup_call_q = quote
            $(Symbol(:warmup_state_,n)) = warmup!(tree, chain, tree_statistics, ϵs, sampling_logdensity, stages[$n], $(Symbol(:warmup_state_,n-1)))
        end
        push!(q.args, warmup_call_q)
    end
    push!(q.args, Symbol(:warmup_state_,N))
    q
end

