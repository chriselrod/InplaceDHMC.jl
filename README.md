# InplaceDHMC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/InplaceDHMC.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriselrod.github.io/InplaceDHMC.jl/dev)
[![Build Status](https://travis-ci.com/chriselrod/InplaceDHMC.jl.svg?branch=master)](https://travis-ci.com/chriselrod/InplaceDHMC.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/InplaceDHMC.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/InplaceDHMC-jl)
[![Codecov](https://codecov.io/gh/chriselrod/InplaceDHMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/InplaceDHMC.jl)
[![Coveralls](https://coveralls.io/repos/github/chriselrod/InplaceDHMC.jl/badge.svg?branch=master)](https://coveralls.io/github/chriselrod/InplaceDHMC.jl?branch=master)
[![Build Status](https://api.cirrus-ci.com/github/chriselrod/InplaceDHMC.jl.svg)](https://cirrus-ci.com/github/chriselrod/InplaceDHMC.jl)


This library began as part of ProbabilityModels, overloading a variety of methods from [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) to improve the interface.
I have now duplicated the library altogether, copying/modifying code as needed. Why? Heavily interfacing with an internal API -- explicitly subject to breaking changes --
is unwise. Additionally, heavy integration allows me to support fixed size arrays and [ProbabilityModels](https://github.com/chriselrod/ProbabilityModels.jl), and reduce memory allocations.

Finally, it allows me to reduce dependencies, by dropping the likes of [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) in favor of the lighter (but much less featureful) QuasiNewtonMethods.jl.

All that said, the contributors to DynamicHMC, as well as those developing the theory of HMC (eg, Betancourt, et al) all deserve much more credit than I.

