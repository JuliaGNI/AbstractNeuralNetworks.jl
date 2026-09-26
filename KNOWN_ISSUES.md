# Known issues

## KI-1 · missing test · `parameterlength(::Dense)` is untested

`src/layers/dense.jl:59` computes the parameter count of a `Dense` layer. A mutant of that line
survives every test file, on `main` and after the test-layout migration.

## KI-2 · docs · the issue reference of the broken Aqua check is not on its line

`test/quality/aqua.jl:7–8`: the comment that names issue #43 is on the line above
`Aqua.test_ambiguities(AbstractNeuralNetworks; broken = true)`. The test-layout rule wants the
issue on the same line as the broken check. Fix: move `# issue #43` onto that line.

## KI-3 · missing test · the HDF5 extension test catches a broken `save` only by errors

`test/integration/hdf5_ext.jl`: a mutant that makes `save(::HDF5.H5DataStore, ::NeuralNetwork)`
write empty parameters is caught by 6 errors (`KeyError: key "L1" not found`) and 2 failures, not by
an assertion that checks what `save` wrote.
