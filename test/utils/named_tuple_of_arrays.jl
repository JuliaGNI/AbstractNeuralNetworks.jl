using AbstractNeuralNetworks
using AbstractNeuralNetworks: NamedTupleOfArrays, ArrayOrNamedTuple
using Random
using Test

Random.seed!(123)

@testset "public, not exported" begin
    for s in (:NamedTupleOfArrays, :ArrayOrNamedTuple)
        @test Base.ispublic(AbstractNeuralNetworks, s)
        @test !Base.isexported(AbstractNeuralNetworks, s)
    end
    @test !Base.ispublic(AbstractNeuralNetworks, :TupleOfArrays)
end

@testset "dispatch" begin
    for T in (Float32, Float64)
        qp = (q = rand(T, 2, 3), p = rand(T, 2, 3))
        @test qp isa NamedTupleOfArrays{T, (:q, :p)}
        @test qp isa ArrayOrNamedTuple{T}
        @test rand(T, 2) isa ArrayOrNamedTuple{T}
        @test (q = view(rand(T, 2, 3), :, 1:2), p = rand(T, 2, 2)) isa
              NamedTupleOfArrays{T, (:q, :p)}
        @test !((q = rand(Float16, 2), p = rand(T, 2)) isa NamedTupleOfArrays)
    end
end
