# The parameter container itself is tested in `NeuralNetworkParameters`. What is tested here is the
# seam: that a network built here carries upstream's container, and that it reaches upstream's tree
# walks and HDF5 path.

using AbstractNeuralNetworks
using AbstractNeuralNetworks: CPUStatic, _statify, networkbackend, save, load
using KernelAbstractions: allocate, ones, zeros
using HDF5
using NeuralNetworkParameters: NetworkParameters, params
using Random
using StaticArrays
using Test

Random.seed!(123)

@testset "0.7 removed the `NeuralNetworkParameters` name from this package" begin
    # 0.6 exported a struct of its own by that name. It is gone — not aliased — so that one type has
    # one name; code written against the old name has to reach for `NetworkParameters` upstream.
    @test !isdefined(AbstractNeuralNetworks, :NeuralNetworkParameters)
    @test :NeuralNetworkParameters ∉ names(AbstractNeuralNetworks)

    # `params` is still ours to import, and still the accessor for the wrapped `NamedTuple`
    @test AbstractNeuralNetworks.params === params
    @test :params ∈ names(AbstractNeuralNetworks)
end

@testset "the container behaves as the struct this package used to define" begin
    nt = (L1 = (W = [1.0 2.0], b = [3.0]),)
    @test NetworkParameters(nt) == NetworkParameters{keys(nt)}(values(nt))
    @test params(NetworkParameters(nt)) === nt
    @test NamedTuple(NetworkParameters(nt)) === nt
end

@testset "a network's parameters are one" begin
    nn = NeuralNetwork(Chain(Dense(4, 3, tanh), Dense(3, 2, tanh)))
    p = params(nn)

    # `initialparameters(::Chain)` builds these with `NetworkParameters{keys}(vals)`, and
    # `NeuralNetwork`'s `PT <: NetworkParameters` bound has to accept the result
    @test p isa NetworkParameters
    @test keys(p) == (:L1, :L2)
    @test size(p.L1.W) == (3, 4)
end

@testset "changebackend walks the tree" begin
    nn = NeuralNetwork(Chain(Dense(4, 3, tanh), Dense(3, 2, tanh)))
    nn_cpu = changebackend(CPU(), nn)

    @test params(nn_cpu) isa NetworkParameters
    @test params(nn_cpu) == params(nn)
    # the `NamedTuple` method of the same walk
    @test changebackend(CPU(), params(nn).L1) == params(nn).L1
    # and a `Tuple` branch inside the tree, which the two hand-written methods this replaced did
    # not reach at all — there was no `Tuple` method, so such a branch was a `MethodError`
    nested = NetworkParameters((L1 = (W = [1.0 2.0], pair = ([3.0], [4.0;;])),))
    @test changebackend(CPU(), nested).L1.pair == nested.L1.pair
end

@testset "the static backend walks the tree" begin
    nn = NeuralNetwork(Chain(Dense(2, 3, tanh), Dense(3, 1, tanh)),
                       AbstractNeuralNetworks.CPUStatic())
    @test params(nn) isa NetworkParameters
    @test params(nn).L1.W isa MArray

    # `_statify` is the same walk, over a dense CPU network's parameters
    static = _statify(params(NeuralNetwork(Chain(Dense(2, 3, tanh)))))
    @test static isa NetworkParameters
    @test static.L1.W isa MArray
end

@testset "the static backend allocates and reports itself" begin
    # the three `KernelAbstractions` methods `CPUStatic` exists to provide, and the `networkbackend`
    # that reads the backend back off a leaf — the round trip `changebackend` below depends on
    @test ones(CPUStatic(), Float64, 2, 3) isa MArray
    @test zeros(CPUStatic(), Float64, 2, 3) isa MArray
    @test size(allocate(CPUStatic(), Float64, 2, 3)) == (2, 3)
    @test allocate(CPUStatic(), Float64, 2, 3) isa MArray

    nn = NeuralNetwork(Chain(Dense(2, 3, tanh)), CPUStatic())
    @test networkbackend(params(nn).L1.W) == CPUStatic()

    # only dense CPU arrays can be made static; a view is an `AbstractArray` that is not an `Array`
    @test_throws ErrorException _statify(view(rand(3, 3), 1:2, 1:2))
end

@testset "a static network walks back to the CPU" begin
    # the return leg of the walk, which reaches `changebackend(::NeuralNetworkBackend, ::MArray)`.
    # `mapparameters` drives both directions, so both belong under test.
    static = NeuralNetwork(Chain(Dense(2, 3, tanh), Dense(3, 1, tanh)), CPUStatic())
    back = changebackend(CPU(), static)

    @test params(back) isa NetworkParameters
    @test params(back).L1.W isa Array
    @test !(params(back).L1.W isa MArray)
    @test params(back).L1.W == params(static).L1.W

    x = rand(2)
    @test back(x) ≈ static(x)
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
            load(NetworkParameters, file)
        end

        @test keys(pread) == keys(p)
        @test pread == p
    finally
        isfile(h5file) && rm(h5file)
    end
end
