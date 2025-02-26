function ZygoteRules.pullback(::typeof(applychain), layers::Tuple, x, params::NeuralNetworkParameters)
    y, pb = ZygoteRules.pullback(applychain, layers, x, values(params))
    function applychain_for_nnps_pullback(output)
        l̄, x̄, p̄ = pb(output)
        l̄, x̄, NeuralNetworkParameters{keys(params)}(p̄)
    end
    y, applychain_for_nnps_pullback
end

function ZygoteRules.pullback(f::Function, params::NeuralNetworkParameters)
    y, pb = ZygoteRules.pullback(f, NamedTuple{keys(params)}(values(params)))
    function gradient_pullback(output)
        p̄ = pb(output)[1]
        (NeuralNetworkParameters{keys(params)}(_values(p̄)),)
    end
    y, gradient_pullback
end

_values(nt::NamedTuple) = values(nt)
_values(nt::NamedTuple{(:params,), Tuple{AT}}) where {AT <: NamedTuple} = _values(nt.params)