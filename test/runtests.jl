using SafeTestsets

@safetestset "Abstract Layer                                                                  " begin include("layers/abstract_layer_tests.jl") end
@safetestset "Dense Layer                                                                     " begin include("layers/dense_layer_tests.jl") end
@safetestset "Linear Layer                                                                    " begin include("layers/linear_layer_tests.jl") end
@safetestset "Chain                                                                           " begin include("layers/chain_tests.jl") end
