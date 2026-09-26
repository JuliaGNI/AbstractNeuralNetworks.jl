using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Utilities" include("utils/utils.jl")
    @safetestset "Abstract Layer" include("layers/abstract.jl")
    @safetestset "Dense Layer" include("layers/dense.jl")
    @safetestset "Linear Layer" include("layers/linear.jl")
    @safetestset "Affine Layer" include("layers/affine.jl")
    @safetestset "Chain" include("chain.jl")
    @safetestset "Neural Network" include("neural_network.jl")
    @safetestset "Neural Network constructors" include("neural_network_constructors.jl")
    @safetestset "Parameters seam" include("integration/parameters_seam.jl")
    @safetestset "Static CPU Backend" include("static_cpu_backend.jl")
    @safetestset "Losses" include("losses.jl")
    @safetestset "Zygote pullback" include("pullback_for_applychain.jl")
    @safetestset "HDF5 save/load" include("integration/hdf5_ext.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
