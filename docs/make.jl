using AbstractNeuralNetworks
using Documenter

DocMeta.setdocmeta!(AbstractNeuralNetworks, :DocTestSetup, :(using AbstractNeuralNetworks); recursive=true)

makedocs(;
    modules=[AbstractNeuralNetworks],
    authors="Michael Kraus",
    repo="https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/blob/{commit}{path}#{line}",
    sitename="AbstractNeuralNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/AbstractNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/AbstractNeuralNetworks.jl",
    devbranch="main",
)
