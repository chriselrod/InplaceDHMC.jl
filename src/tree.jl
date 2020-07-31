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
struct FlaggedVector{D,T,L} <: PaddedMatrices.AbstractMutableFixedSizeVector{D,T,L}
    v::PtrVector{D,T,1,0,0,false}
    flag::UInt32
end
@inline function FlaggedVector{D,T,L}(ptr::Ptr{T}, flag::UInt32) where {D,T,L}
    FlaggedVector{D,T,L}(PtrVector{D,T,1,0,0,false}(ptr), flag)
end
@inline Base.pointer(v::FlaggedVector) = v.v.ptr

function Tree{D,T,L}(sptr::StackPointer, depth::Int = DEFAULT_MAX_TREE_DEPTH) where {D,T,L}
    root = pointer(sptr, T)
    depth₊ = depth + 3
    # set roots to zero so that all loads can properly be interpreted as bools without further processing on future accesses.
    tree = Tree{D,T,L}( root, depth₊, sptr + VectorizationBase.REGISTER_SIZE + 6L*sizeof(T)*depth₊ )
    clear!(tree)
    tree
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
    flag != 0x00000000 && @assert (allocated | flag) != allocated "Double free!!!!"
    VectorizationBase.store!(root32, VectorizationBase.load(root32) | flag)
    nothing
end
free_z!(tree::Tree, flag::UInt32) = free!(tree, flag, 0)
free_ρ♯!(tree::Tree, flag::UInt32) = free!(tree, flag, 8)
free_Σρ!(tree::Tree, flag::UInt32) = free!(tree, flag, 4)

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
    ζ = combine_proposals(rng, tree, trajectory, ζ₁, ζ₂, logprob2, is_forward)#, is_doubling)
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
function adjacent_tree(rng, tree::Tree{P,T,L}, H, trajectory, z::PhasePoint{P,T,L}, i::Int32, depth::Int32, is_forward::Bool) where {P,T,L}
    i′ = i + (is_forward ? one(Int32) : -one(Int32) )
    # @show (1, bitstring(unsafe_load(reinterpret(Ptr{UInt32}, tree.root), 2)))
    # @show z.Q.q
    # @show depth, i′
    lb, ub = 5, 10
    # lb <= abs(i′) < ub && @show z
    if depth == zero(Int32) # moves from z into ζready
        z′ = move(tree, H, trajectory, z, is_forward)
        # lb <= abs(i′) < ub && @show logdensity(trajectory.H, z′), trajectory.π₀
        (ζ, ω, τ), v, invalid = leaf(tree, H, trajectory, z′, false)
        return (ζ, ω, τ, z′, i′), v, (invalid,InvalidTree(i′))
    else
        # “left” tree
        t₋, v₋, (invalid,it) = adjacent_tree(
            rng, tree, H, trajectory, z, i, depth - one(Int32), is_forward
        )
        # @show first(t₋)
        # @show t₋[4]
        invalid && return t₋, v₋, (invalid, it)
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        # “right” tree — visited information from left is kept even if invalid
        t₊, v₊, (invalid,it) = adjacent_tree(
            rng, tree, H, trajectory, z₋, i₋, depth - one(Int32), is_forward
        )
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
        ζ, ω = combine_proposals_and_logweights(
            rng, tree, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false
        )
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
function sample_trajectory(rng, tree::Tree, H, trajectory, z::PhasePoint{P,T,L}, max_depth::Integer, directions::Directions) where {P,T,L}
    #    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    # original_flag = zᵢ.flag
    # protect_initial = true # Protect initial position by giving it a dummy flag
    # z = PhasePoint(zᵢ.Q, zᵢ.p, 0x00000000)
    # @show logdensity(trajectory.H, z), trajectory.π₀
    (ζ, ω, τ), v, invalid = leaf(tree, H, trajectory, z, true)
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
        VectorizationBase.store!(
            reinterpret(Ptr{UInt32}, tree.root), 0xffffffff ⊻ (alloc | ζ.flag)
        )
        # clear_all_but_z!( tree, alloc | ζ.flag )
        t′, v′, (invalid, it) = adjacent_tree(
            rng, tree, H, trajectory, zᵢ, iᵢ, depth, is_forward
        )
        
        v = combine_visited_statistics(trajectory, v, v′)

        # invalid adjacent tree: stop
        invalid && (termination = it; break)

        # extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′
        # allocate!(tree, z′.flag)
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


