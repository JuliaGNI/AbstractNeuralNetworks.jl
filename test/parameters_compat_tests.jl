# The parameter container itself is tested in `NeuralNetworkParameters`. What is tested here is the
# seam: that the compatibility alias behaves as the struct this package used to define, and that a
# network built here reaches upstream's tree walks and HDF5 path.

using AbstractNeuralNetworks
using AbstractNeuralNetworks: NeuralNetworkParameters, _statify, save, load
using HDF5
using Random
using StaticArrays
using Test

import NeuralNetworkParameters as NNP

Random.seed!(123)

@testset "the alias is the upstream type" begin
    # the same object, not a second definition — which is what makes `::Type{}` dispatch, `<:` bounds
    # and `{keys}(vals)` construction work unchanged for code written against either name
    @test NeuralNetworkParameters === NNP.NetworkParameters

    nt = (L1 = (W = [1.0 2.0], b = [3.0]),)
    @test NeuralNetworkParameters(nt) == NeuralNetworkParameters{keys(nt)}(values(nt))
    @test params(NeuralNetworkParameters(nt)) === nt
    @test NamedTuple(NeuralNetworkParameters(nt)) === nt
end

@testset "a network's parameters are one" begin
    nn = NeuralNetwork(Chain(Dense(4, 3, tanh), Dense(3, 2, tanh)))
    p = params(nn)

    # `initialparameters(::Chain)` builds these with `NeuralNetworkParameters{keys}(vals)`, and
    # `NeuralNetwork`'s `PT <: NeuralNetworkParameters` bound has to accept the result
    @test p isa NeuralNetworkParameters
    @test keys(p) == (:L1, :L2)
    @test size(p.L1.W) == (3, 4)
end

@testset "changebackend walks the tree" begin
    nn = NeuralNetwork(Chain(Dense(4, 3, tanh), Dense(3, 2, tanh)))
    nn_cpu = changebackend(CPU(), nn)

    @test params(nn_cpu) isa NeuralNetworkParameters
    @test params(nn_cpu) == params(nn)
    # the `NamedTuple` method of the same walk
    @test changebackend(CPU(), params(nn).L1) == params(nn).L1
end

@testset "the static backend walks the tree" begin
    nn = NeuralNetwork(Chain(Dense(2, 3, tanh), Dense(3, 1, tanh)),
                       AbstractNeuralNetworks.CPUStatic())
    @test params(nn) isa NeuralNetworkParameters
    @test params(nn).L1.W isa MArray

    # `_statify` is the same walk, over a dense CPU network's parameters
    static = _statify(params(NeuralNetwork(Chain(Dense(2, 3, tanh)))))
    @test static isa NeuralNetworkParameters
    @test static.L1.W isa MArray
end

@testset "HDF5 round trip keeps the layer order" begin
    # ten layers is the point: HDF5 hands group members back sorted, so `L10` precedes `L2` unless
    # the writer records the key order. This package's own extension did not and could not pass this
    # test; upstream's writes a `keys` attribute.
    nn = NeuralNetwork(Chain(Dense(4, 4, tanh),
                             Tuple(Dense(4, 4, tanh) for _ in 1:8)...,
                             Dense(4, 4, tanh)))
    p = params(nn)
    @test length(keys(p)) == 10

    h5file = tempname() * ".h5"
    try
        h5open(h5file, "w") do file
            save(file, p)
        end
        pread = h5open(h5file, "r") do file
            load(NeuralNetworkParameters, file)
        end

        @test keys(pread) == keys(p)
        @test pread == p
    finally
        isfile(h5file) && rm(h5file)
    end
end
