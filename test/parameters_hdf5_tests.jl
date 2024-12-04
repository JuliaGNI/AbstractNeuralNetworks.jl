using AbstractNeuralNetworks
using AbstractNeuralNetworks: params, load, save
using HDF5
using Random
using Test

h5file = "temp.h5"

Random.seed!(123)

c = Chain(Dense(4, 4, x -> x),
          Dense(4, 4, x -> x),
          Dense(4, 4, x -> x))
n = NeuralNetwork(c, Float64; initializer = GlorotUniform())
p = params(n)


h5open(h5file, "w") do file
    save(file, p)
end

@test isfile(h5file)

pread = h5open(h5file, "r") do file
    load(NeuralNetworkParameters, file)
end

rm(h5file)

@test p == pread
