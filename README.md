# InplaceDHMC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/InplaceDHMC.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriselrod.github.io/InplaceDHMC.jl/dev)
[![CI](https://github.com/chriselrod/InplaceDHMC.jl/workflows/CI/badge.svg)](https://github.com/chriselrod/InplaceDHMC.jl/actions?query=workflow%3ACI)
[![CI (Julia nightly)](https://github.com/chriselrod/InplaceDHMC.jl/workflows/CI%20(Julia%20nightly)/badge.svg)](https://github.com/chriselrod/InplaceDHMC.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22)
[![Codecov](https://codecov.io/gh/chriselrod/InplaceDHMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/InplaceDHMC.jl)
[![Build Status](https://api.cirrus-ci.com/github/chriselrod/InplaceDHMC.jl.svg)](https://cirrus-ci.com/github/chriselrod/InplaceDHMC.jl)


This library began as part of ProbabilityModels, overloading a variety of methods from [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) to improve the interface.
I have now duplicated the library altogether, copying/modifying code as needed. Why? Heavily interfacing with an internal API -- explicitly subject to breaking changes --
is unwise. Additionally, heavy integration allows me to support fixed size arrays and [ProbabilityModels](https://github.com/chriselrod/ProbabilityModels.jl), and reduce memory allocations.

Finally, it allows me to reduce dependencies, by dropping the likes of [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) in favor of the lighter (but much less featureful) QuasiNewtonMethods.jl.

All that said, the contributors to DynamicHMC, as well as those developing the theory of HMC (eg, Betancourt, et al) all deserve much more credit than I.

