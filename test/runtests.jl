using SafeTestsets

@safetestset "Utilities                                                                       " begin include("utils_tests.jl") end
@safetestset "Abstract Layer                                                                  " begin include("layers/abstract_layer_tests.jl") end
@safetestset "Dense Layer                                                                     " begin include("layers/dense_layer_tests.jl") end
@safetestset "Linear Layer                                                                    " begin include("layers/linear_layer_tests.jl") end
@safetestset "Affine Layer                                                                    " begin include("layers/affine_layer_tests.jl") end
@safetestset "Parameters                                                                      " begin include("parameters_tests.jl") end
@safetestset "Chain                                                                           " begin include("chain_tests.jl") end
@safetestset "Architecture                                                                    " begin include("architecture_tests.jl") end
@safetestset "Neural Network                                                                  " begin include("neural_network_tests.jl") end
@safetestset "Neural Network constructors                                                     " begin include("neural_network_constructors.jl") end 
@safetestset "Parameters HDF5 Routines                                                        " begin include("parameters_hdf5_tests.jl") end
@safetestset "Static CPU Backend                                                              " begin include("static_backend.jl")

# @safetestset "Identity Cell                                                                   " begin include("cells/identity_tests.jl") end
# @safetestset "Recurrent Cell                                                                  " begin include("cells/recurrent_tests.jl") end
# @safetestset "GRU Cell                                                                        " begin include("cells/gru_tests.jl") end
# @safetestset "LSTM Cell                                                                       " begin include("cells/lstm_tests.jl") end
# @safetestset "Grid Cell                                                                       " begin include("cells/grid_tests.jl") end
