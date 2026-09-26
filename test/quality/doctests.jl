using Documenter
using AbstractNeuralNetworks
using Test

# Documenter evaluates a page's `@meta` block in `Main`, and this file runs in a module of its own.
@eval Main import AbstractNeuralNetworks

DocMeta.setdocmeta!(
    AbstractNeuralNetworks, :DocTestSetup, :(using AbstractNeuralNetworks);
    recursive = true)

doctest(AbstractNeuralNetworks)
