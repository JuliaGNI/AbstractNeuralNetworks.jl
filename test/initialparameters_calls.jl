using AbstractNeuralNetworks 
using Test 
import Random 

function test_different_cpu_initializations(::Type{T}) where T <: Number
    model = Chain(Dense(4, 5, tanh), Linear(5, 4))
    Random.seed!(123)
    ps1  = initialparameters(model, CPU(), T)
    Random.seed!(123)
    ps2  = initialparameters(model, T)
    Random.seed!(123)
    ps3  = T == Float64 ? initialparameters(model, CPU()) : ps1
    Random.seed!(123)
    ps4 = T == Float64 ? initialparameters(model) : ps2

    @test ps1 == ps2 == ps3 == ps4
end

test_different_cpu_initializations(Float32)
test_different_cpu_initializations(Float64)
test_different_cpu_initializations(Float16)