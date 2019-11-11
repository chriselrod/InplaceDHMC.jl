
function threadrandinit!(sptr::StackPointer, pcg_vector::Vector{PtrPCG{P}}) where {P}
    W = VectorizationBase.pick_vector_width(Float64)
#    myprocid = 0#myid()-1
    local_stack_size = LOCAL_STACK_SIZE[]
    stack_ptr = sptr
    for n ∈ 1:NTHREADS[]
        rng = VectorizedRNG.random_init_pcg!(PtrPCG{P}(stack_ptr))#, P*(n-1)*myprocid)
        if n > length(pcg_vector)
            push!(pcg_vector, rng)
        else
            pcg_vector[n] = rng
        end
        stack_ptr += local_stack_size
    end
    sptr + (2P+1)*8W
end

