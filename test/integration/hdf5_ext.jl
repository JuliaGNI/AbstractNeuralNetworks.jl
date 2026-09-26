using AbstractNeuralNetworks
using HDF5
using Random
using Test

import AbstractNeuralNetworks: params

# `TestArchitecture` stands in for a concrete architecture: `load(::Type{NeuralNetwork}, …)`
# rebuilds the model through `Chain(arch)`, which every real architecture in this ecosystem defines.
struct TestArchitecture <: AbstractNeuralNetworks.Architecture end
AbstractNeuralNetworks.Chain(::TestArchitecture) = Chain(Dense(4, 4, tanh), Linear(4, 4))

@testset "save/load roundtrip ($T)" for T in (Float32, Float64)
    arch = TestArchitecture()
    Random.seed!(1)
    nn = NeuralNetwork(arch, CPU(), T)
    x = rand(T, 4)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "nn.h5")
        AbstractNeuralNetworks.save(path, nn)
        nn2 = AbstractNeuralNetworks.load(NeuralNetwork, path, arch)

        @test params(nn) == params(nn2)
        @test nn2(x) ≈ y_before
        @test eltype(params(nn2)[1].W) == T
    end
end

@testset "save/load via an open H5DataStore" begin
    arch = TestArchitecture()
    Random.seed!(2)
    nn = NeuralNetwork(arch, CPU(), Float64)
    x = rand(4)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "nn_store.h5")
        HDF5.h5open(path, "w") do h5
            AbstractNeuralNetworks.save(h5, nn)
        end
        nn2 = HDF5.h5open(path, "r") do h5
            AbstractNeuralNetworks.load(NeuralNetwork, h5, arch)
        end
        @test nn2(x) ≈ y_before
    end
end

@testset "save/load against a prototype parameter set" begin
    arch = TestArchitecture()
    Random.seed!(3)
    nn = NeuralNetwork(arch, CPU(), Float64)
    x = rand(4)
    y_before = nn(x)
    prototype = params(NeuralNetwork(arch, CPU(), Float64))

    mktempdir() do dir
        path = joinpath(dir, "nn_prototype.h5")
        AbstractNeuralNetworks.save(path, nn)
        nn2 = AbstractNeuralNetworks.load(NeuralNetwork, path, arch, prototype)

        @test params(nn) == params(nn2)
        @test nn2(x) ≈ y_before
    end
end
