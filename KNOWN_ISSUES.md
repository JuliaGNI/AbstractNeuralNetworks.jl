# Known issues

### K1 · No test covers `parameterlength(::Dense)`.

- **location:** `src/layers/dense.jl:59`
- **evidence:** the line computes the parameter count of a `Dense` layer. A mutant of that line
  survives every test file.
- **kind:** missing test
- **found:** 2026-09-26

### K2 · The HDF5 extension test catches a broken `save` only by errors.

- **location:** `test/integration/hdf5_ext.jl`
- **evidence:** a mutant that makes `save(::HDF5.H5DataStore, ::NeuralNetwork)` write empty
  parameters is caught by 6 errors (`KeyError: key "L1" not found`) and 2 failures, not by an
  assertion that checks what `save` wrote.
- **kind:** missing test
- **found:** 2026-09-26

### K3 · A comment in `src/chain.jl` names the test file `test/chain_tests.jl`, which is `test/chain.jl`.

- **location:** `src/chain.jl:59`
- **evidence:** the comment above `applychain(layers::Tuple, x, ps::NamedTuple)` reads "Without
  this method, the test in `test/chain_tests.jl` that calls a `Chain` with a bare `NamedTuple`
  fails". `git ls-files test/chain_tests.jl` prints nothing; the test is in `test/chain.jl`.
- **kind:** docs
- **found:** 2026-09-26
