function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Base.indexed_iterate),Tuple{DynamicPaddedArray{Float64,2},DynamicPaddedArray{InplaceDHMC.TreeStatisticsNUTS,1}},Int64})
    precompile(Tuple{typeof(Base.indexed_iterate),Tuple{DynamicPaddedArray{Float64,3},DynamicPaddedArray{InplaceDHMC.TreeStatisticsNUTS,2}},Int64})
    precompile(Tuple{typeof(InplaceDHMC.regularized_cov_block_quote),Int64,Type{T} where T,Int64,Int64,Bool})
    precompile(Tuple{typeof(VectorizationBase.T_shift),Type{InplaceDHMC.TreeStatisticsNUTS}})
end
