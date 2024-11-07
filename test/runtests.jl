using SafeTestsets

@safetestset "Abstract Layer                                                                  " begin include("layers/abstract_layer_tests.jl") end
@safetestset "Dense Layer                                                                     " begin include("layers/dense_layer_tests.jl") end
@safetestset "Linear Layer                                                                    " begin include("layers/linear_layer_tests.jl") end
@safetestset "Chain                                                                           " begin include("chain_tests.jl") end
@safetestset "Architecture                                                                    " begin include("architecture_tests.jl") end
@safetestset "Neural Network                                                                  " begin include("neural_network_tests.jl") end
@safetestset "Neural Network constructors                                                     " begin include("neural_network_constructors.jl") end 
@safetestset "Initialparameters calls                                                         " begin include("initialparameters_calls.jl") end

@safetestset "Identity Cell                                                                   " begin include("cells/identity_tests.jl") end
@safetestset "Recurrent Cell                                                                  " begin include("cells/recurrent_tests.jl") end
@safetestset "GRU Cell                                                                        " begin include("cells/gru_tests.jl") end
@safetestset "LSTM Cell                                                                       " begin include("cells/lstm_tests.jl") end
@safetestset "GridCell                                                                        " begin include("cells/grid_tests.jl") end
