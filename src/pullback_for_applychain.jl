# The gradient of a `NetworkParameters` holds the gradient with respect to the storage of each leaf,
# as `NeuralNetworkParameters` gives it for a loss of a set. Zygote gives the natural cotangent of each
# leaf of `values(params)`, and `map_cotangent` converts it with `storage_gradient`.
function ZygoteRules.pullback(::typeof(applychain), layers::Tuple, x, params::NetworkParameters)
    y, pb = ZygoteRules.pullback(applychain, layers, x, values(params))
    function applychain_for_nnps_pullback(output)
        l̄, x̄, p̄ = pb(output)
        p̄ = map_cotangent(storage_gradient, values(params), p̄)
        l̄, x̄, p̄ === nothing ? nothing : NetworkParameters(NamedTuple{keys(params)}(p̄))
    end
    y, applychain_for_nnps_pullback
end

# The generic `ZygoteRules.pullback(f::Function, ::NetworkParameters)` used to live here too. It
# belongs to the `NeuralNetworkParameters` package now (`ext/ZygoteRulesExt.jl` there): with the type
# defined upstream, `ZygoteRules.pullback` was the only name in that signature this package owned,
# and a method needs to own just one of them. The method above stays, because `applychain` is this
# package's.
