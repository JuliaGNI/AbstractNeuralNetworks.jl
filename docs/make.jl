using AbstractNeuralNetworks
using Documenter
using DocumenterCitations
import Pkg

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "github.com/JuliaGNI/AbstractNeuralNetworks.jl"

bib = CitationBibliography(joinpath(@__DIR__, "src", "AbstractNeuralNetworks.bib"))

DocMeta.setdocmeta!(AbstractNeuralNetworks, :DocTestSetup, :(using AbstractNeuralNetworks); recursive=true)

makedocs(;
    plugins=[bib],
    modules=[AbstractNeuralNetworks],
    authors=AUTHORS,
    repo="https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/blob/{commit}{path}#{line}",
    sitename=NAME,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/AbstractNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Static Neural Network Parameters" => "static_neural_network_parameters.md",
        "References" => "bibliography.md"
    ],
)

deploydocs(;
    repo   = GITHUB,
    devurl = "latest",
    devbranch = "main",
)
