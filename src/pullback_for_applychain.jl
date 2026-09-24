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
