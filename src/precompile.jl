function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{InplaceDHMC.var"##s18#32",Any,Any,Any,Any,Any,Any,Any,Any})
    precompile(Tuple{typeof(InplaceDHMC.regularized_cov_block_quote),Int64,Type,Int64,Int64,Bool})
end
