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

# """
# $(SIGNATURES)
# Perform MCMC with NUTS, keeping the warmup results. Returns a `NamedTuple` of
# - `initial_warmup_state`, which contains the initial warmup state
# - `warmup`, an iterable of `NamedTuple`s each containing fields
#     - `stage`: the relevant warmup stage
#     - `results`: results returned by that warmup stage (may be `nothing` if not applicable,
#       or a chain, with tree statistics, etc; see the documentation of stages)
#     - `warmup_state`: the warmup state *after* the corresponding stage.
# - `final_warmup_state`, which contains the final adaptation after all the warmup
# - `inference`, which has `chain` and `tree_statistics`, see [`mcmc_with_warmup`](@ref).
# !!! warning
#     This function is not (yet) exported because the the warmup interface may change with
#     minor versions without being considered breaking. Recommended for interactive use.
# $(DOC_MCMC_ARGS)
# """
# function mcmc_keep_warmup(rng::AbstractRNG, tree::Tree, ℓ, N::Integer;
#                           initialization = (),
#                           warmup_stages = default_warmup_stages(),
#                           algorithm = NUTS(),
#                           reporter = default_reporter())
#     sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
#     sptr, initial_warmup_state = initialize_warmup_state(rng, sptr, ℓ; initialization...)
#     warmup, warmup_state = _warmup(sampling_logdensity, warmup_stages, initial_warmup_state)
#     inference = mcmc(sampling_logdensity, N, warmup_state)
#     (initial_warmup_state = initial_warmup_state, warmup = warmup,
#      final_warmup_state = warmup_state, inference = inference)
# end

# """
# $(SIGNATURES)
# Perform MCMC with NUTS, including warmup which is not returned. Return a `NamedTuple` of
# - `chain`, a vector of positions from the posterior
# - `tree_statistics`, a vector of tree statistics
# - `κ` and `ϵ`, the adapted metric and stepsize.
# $(DOC_MCMC_ARGS)
# # Usage examples
# Using a fixed stepsize:
# ```julia
# mcmc_with_warmup(rng, ℓ, N;
#                  initialization = (ϵ = 0.1, ),
#                  warmup_stages = fixed_stepsize_warmup_stages())
# ```
# Starting from a given position `q₀` and kinetic energy scaled down (will still be adapted):
# ```julia
# mcmc_with_warmup(rng, ℓ, N;
#                  initialization = (q = q₀, κ = GaussianKineticEnergy(5, 0.1)))
# ```
# Using a dense metric:
# ```julia
# mcmc_with_warmup(rng, ℓ, N;
#                  warmup_stages = default_warmup_stages(; M = Symmetric))
# ```
# Disabling the optimization step:
# ```julia
# mcmc_with_warmup(rng, ℓ, N;
#                  warmup_stages = default_warmup_stages(; local_optimization = nothing,
#                                                          M = Symmetric))
# ```
# """
# function mcmc_with_warmup(rng, ℓ, N; initialization = (),
#                           warmup_stages = default_warmup_stages(),
#                           algorithm = NUTS(), reporter = default_reporter())
#     @unpack final_warmup_state, inference =
#         mcmc_keep_warmup(rng, ℓ, N; initialization = initialization,
#                          warmup_stages = warmup_stages, algorithm = algorithm,
#                          reporter = reporter)
#     @unpack κ, ϵ = final_warmup_state
#     (inference..., κ = κ, ϵ = ϵ)
# end

function mcmc_with_warmup!(
    rng::AbstractRNG, sptr::StackPointer, chain::AbstractMatrix{T}, tree_statistics::AbstractVector{TreeStatisticsNUTS}, ℓ::AbstractProbabilityModel{D}, N::Int = size(chain,2);
    initialization = (), warmup_stages = default_warmup_stages(), algorithm = NUTS(), reporter = default_reporter()
) where {D,T}

    # We allocate the tree here.
    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    sptr, ϵs = DynamicPtrVector{Float64}(sptr, maximum(length, warmup_stages))
    tree, initial_warmup_state = initialize_warmup_state(rng, sptr, ℓ; initialization...)
    warmup_state = _warmup!(tree, chain, tree_statistics, ϵs, sampling_logdensity, warmup_stages, initial_warmup_state)
    mcmc!(tree, chain, tree_statistics, sampling_logdensity, N, warmup_state)

end


function mcmc_with_warmup(
    ℓ::AbstractProbabilityModel{D}, N; δ::Float64 = 0.8, initialization = (),
    warmup_stages = default_warmup_stages(stepsize_adaptation=DualAveraging(δ=δ)),
    algorithm = NUTS(), reporter = default_reporter()
) where {D}
    sptr = stackpointer()
    nwarmup = maximum(length, warmup_stages)
    NS = max(N, nwarmup)
    L = VectorizationBase.align(D, Float64)
    chain = DynamicPaddedMatrix{Float64}(undef, (D, NS), L)
    tree_statistics = DynamicPaddedVector{TreeStatisticsNUTS}(undef, NS)
    
    chain_m = DynamicPtrMatrix{Float64}(pointer(chain), (D, NS), L)
    tree_statistic = DynamicPtrVector{TreeStatisticsNUTS}(pointer(tree_statistics), NS)
    mcmc_with_warmup!(
        first(GLOBAL_PCGs), sptr, chain_m, tree_statistic, ℓ, N;
        initialization = initialization, warmup_stages = warmup_stages, algorithm = algorithm, reporter = reporter
    )
    chain, tree_statistics
end

function threaded_mcmc(
    ℓ::AbstractProbabilityModel{D}, N; δ::Float64 = 0.8, initialization = (),
    warmup_stages = default_warmup_stages(stepsize_adaptation=DualAveraging(δ=δ)),
    algorithm = NUTS(), reporter = NoProgressReport(),
    nchains = NTHREADS[]
) where {D}
    sptr = STACK_POINTER_REF[]
    LSS = LOCAL_STACK_SIZE[]
    nwarmup = maximum(length, warmup_stages)

    NS = max(N,nwarmup)
    L = VectorizationBase.align(D, Float64)
    NSA = VectorizationBase.align(NS, Float64)
    chains = DynamicPaddedArray{Float64}(undef, (D, NS, nchains), L)
    tree_stride = NSA + VectorizationBase.CACHELINE_SIZE >> 3
    tree_statistics = DynamicPaddedMatrix{TreeStatisticsNUTS}(undef, (NS, nchains), tree_stride)

    chain_ptr = pointer(chains)
    stat_ptr = pointer(tree_statistics)
    
    Threads.@threads for t in 0:nchains-1
        chain = DynamicPtrMatrix{Float64}(chain_ptr + t*8NS*L, (D, NS), L)
        tree_statistic = DynamicPtrVector{TreeStatisticsNUTS}(stat_ptr + t*tree_stride*sizeof(TreeStatisticsNUTS), (NS,), NSA)
        mcmc_with_warmup!(
            GLOBAL_PCGs[t+1], sptr + t*LSS, chain, tree_statistic, ℓ, N;
            initialization = initialization, warmup_stages = warmup_stages, algorithm = algorithm, reporter = reporter
        )
    end
    chains, tree_statistics
end



