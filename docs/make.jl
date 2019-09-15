using Documenter, InplaceDHMC

makedocs(;
    modules=[InplaceDHMC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/InplaceDHMC.jl/blob/{commit}{path}#L{line}",
    sitename="InplaceDHMC.jl",
    authors="Chris Elrod <elrodc@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/chriselrod/InplaceDHMC.jl",
)
