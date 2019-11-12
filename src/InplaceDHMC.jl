module InplaceDHMC

export
    # kinetic energy
    GaussianKineticEnergy,
    # NUTS
    TreeOptionsNUTS,
    # reporting
    NoProgressReport, LogProgressReport,
    # mcmc
    TuningNUTS, mcmc_with_warmup, threaded_mcmc, default_warmup_stages

using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using LinearAlgebra
using LinearAlgebra: Diagonal, Symmetric
# using LogDensityProblems: capabilities, LogDensityOrder, dimension, logdensity_and_gradient
# import NLSolversBase, Optim # optimization step in mcmc
using Parameters: @unpack
using Random
using Statistics: cov, mean, median, middle, quantile, var

using VectorizedRNG: AbstractPCG, PtrPCG
using VectorizationBase, LoopVectorization, VectorizedRNG, StackPointers, PaddedMatrices, QuasiNewtonMethods, SIMDPirates, Mmap
using QuasiNewtonMethods: AbstractProbabilityModel, dimension, logdensity, logdensity_and_gradient!
using LoopVectorization: @vvectorize_unsafe

# copy from StatsFuns.jl
function logaddexp(x, y)
    isfinite(x) && isfinite(y) || return x > y ? x : y # QuasiNewtonMethods.nanmax(x,y)
    x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
end

const MMAP = Ref{Matrix{UInt8}}()
const STACK_POINTER_REF = Ref{StackPointer}()
const LOCAL_STACK_SIZE = Ref{Int}()
const GLOBAL_PCGs = Vector{PtrPCG{4}}(undef,0)
const NTHREADS = Ref{Int}()

include("hamiltonian.jl")
include("tree.jl")
include("kinetic_energy.jl")
include("stepsize.jl")
include("NUTS.jl")
include("reporting.jl")
include("warmup.jl")
include("mcmc.jl")
include("rng.jl")
# include("diagnostics.jl")

function __init__()
    NTHREADS[] = Threads.nthreads()
    # Note that 1 GiB == 2^30 == 1 << 30 bytesy
    # Allocates 0.5 GiB per thread for the stack by default.
    # Can be controlled via the environmental variable PROBABILITY_MODELS_STACK_SIZE
    LOCAL_STACK_SIZE[] = if "PROBABILITY_MODELS_STACK_SIZE" ∈ keys(ENV)
        parse(Int, ENV["PROBABILITY_MODELS_STACK_SIZE"])
    else
        1 << 30
    end + VectorizationBase.REGISTER_SIZE - 1 # so we have at least the indicated stack size after REGISTER_SIZE-alignment
    MMAP[] = Mmap.mmap(Matrix{UInt8}, LOCAL_STACK_SIZE[], NTHREADS[])
    STACK_POINTER_REF[] = StackPointer(
        VectorizationBase.align(Base.unsafe_convert(Ptr{Cvoid}, pointer(MMAP[])))
    )
    STACK_POINTER_REF[] = threadrandinit!(STACK_POINTER_REF[], GLOBAL_PCGs)
end
function realloc_stack(new_local_stack_size::Integer)
    @warn """You must redefine all probability models. The stack pointers get dereferenced at compile time, and the stack has just been reallocated.
Re-evaluating densities without first recompiling them will likely crash Julia!"""
    LOCAL_STACK_SIZE[] = new_local_stack_size
    MMAP[] = Mmap.mmap(Matrix{UInt8}, LOCAL_STACK_SIZE[], NTHREADS[])
    STACK_POINTER_REF[] = PaddedMatrices.StackPointer(
        VectorizationBase.align(Base.unsafe_convert(Ptr{Cvoid}, pointer(MMAP[])))
    )
    STACK_POINTER_REF[] = threadrandinit!(STACK_POINTER_REF[], GLOBAL_PCGs)
end
stackpointer() = STACK_POINTER_REF[] + (Threads.threadid() - 1) * LOCAL_STACK_SIZE[]


end # module
