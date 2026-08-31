using SafeTestsets

@safetestset "Utilities                                                                       " begin
    include("utils_tests.jl")
end
@safetestset "Abstract Layer                                                                  " begin
    include("layers/abstract_layer_tests.jl")
end
@safetestset "Dense Layer                                                                     " begin
    include("layers/dense_layer_tests.jl")
end
@safetestset "Linear Layer                                                                    " begin
    include("layers/linear_layer_tests.jl")
end
@safetestset "Affine Layer                                                                    " begin
    include("layers/affine_layer_tests.jl")
end
@safetestset "Chain                                                                           " begin
    include("chain_tests.jl")
end
@safetestset "Architecture                                                                    " begin
    include("architecture_tests.jl")
end
@safetestset "Neural Network                                                                  " begin
    include("neural_network_tests.jl")
end
@safetestset "Neural Network constructors                                                     " begin
    include("neural_network_constructors.jl")
end
@safetestset "Parameters seam                                                                 " begin
    include("parameters_seam_tests.jl")
end
@safetestset "Static CPU Backend                                                              " begin
    include("static_backend.jl")
end
@safetestset "Losses                                                                          " begin
    include("losses_tests.jl")
end
@safetestset "Zygote pullback                                                                 " begin
    include("custom_pullback_test.jl")
end
