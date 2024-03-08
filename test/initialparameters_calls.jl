using AbstractNeuralNetworks 
using Test 
import Random 

function test_different_cpu_initializations(T₁::Type{T}) where T <: Number
    model = Chain(Dense(4, 5, tanh), Linear(5, 4))
    Random.seed!(123)
    ps₁  = initialparameters(model, T₁, CPU())
    Random.seed!(123)
    ps₂  = initialparameters(model, T₁)
    Random.seed!(123)
    ps₃  = initialparameters(CPU(), T₁, model)
    Random.seed!(123)
    ps₄  = initialparameters(CPU(), model, T₁)
    Random.seed!(123)
    ps₅  = initialparameters(T₁, CPU(), model)
    Random.seed!(123)
    ps₆  = initialparameters(T₁, model, CPU())
    Random.seed!(123)
    ps₇  = T₁ == Float64 ? initialparameters(model, CPU(), T₁) : ps₆
    Random.seed!(123)
    ps₈  = initialparameters(T₁, model)
    Random.seed!(123)
    ps₉  = T₁ == Float64 ? initialparameters(CPU(), model) : ps₆
    Random.seed!(123)
    ps₁₀ = T₁ == Float64 ? initialparameters(model, CPU()) : ps₆
    Random.seed!(123)
    ps₁₁ = T₁ == Float64 ? initialparameters(model) : ps₈

    @test ps₁ == ps₂ == ps₃ == ps₄ == ps₅ == ps₆ == ps₇ == ps₈ == ps₉ == ps₁₀ == ps₁₁
end

test_different_cpu_initializations(Float32)
test_different_cpu_initializations(Float64)
test_different_cpu_initializations(Float16)