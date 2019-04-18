# CompOpt-LowRankMatrixRecovery
Implementation of nonsmooth optimization algorithms for matrix recovery problems.

Subgradient and prox-linear methods as well as methods for generating
synthetic instances of matrix sensing, robust PCA and matrix completion
instances are available under `src/CompOpt.jl`.

### Dependencies
The scripts depend on `ArgParse` and `PyPlot`. The core library has no external
dependencies.


### Quick Tour
The matrix recovery problems covered are bilinear and quadratic sensing, matrix
completion, and robust PCA. We provide generators for synthetic instances of
all these problems, as demonstrated below:

```julia
include("src/CompOpt.jl");

prob_bilin = CompOpt.genBilinProb(50, 100, 5, 0.25)
prob_quad = CompOpt.genSymQuadProb(50, 5, 0.25)
prob_rpca = CompOpt.genRpcaProb(50, 5, 0.25)
prob_mcomp = CompOpt.genMatCompProb(50, 5, 0.25)
```

In all but the first constructors above, the first argument is the dimension
`d` (number of rows of the matrix), second argument is the rank `r`, and third
argument is either the level of corruption with arbitrary noise (in the case of
matrix sensing and robust PCA) or the fraction of observed entries (in matrix
completion). In the bilinear sensing constructor, the first two arguments are
devoted to sizes of different matrices, since the signals might have different
sizes.

The above constructors are not the only way to generate problem instances. You
can define your own signals and corruptions. Below we give an example of a `100
x 5` quadratic sensing instance with a total of `1000` measurements,
the first 100 of which are corrupted by replacing them with a value chosen in `{-1, 1}` at random.

```julia
using LinearAlgebra
using Random
include("src/CompOpt.jl")

# generate data
d = 100; r = 5; m = 1000; pfail = 0.1
A = randn(m, d); X = randn(d, r)
y = mapslices(x -> norm(x)^2, A * X, dims=[2])[:]  # cast to vector

# corrupt measurements
num_corrupted = trunc(Int, pfail * m)
y[1:num_corrupted] = rand([-1, 1], num_corrupted)

prob = CompOpt.QuadProb(y, X, A, pfail)
```

We have implemented 2 composite optimization methods - subgradient and prox-linear - which are available
as the functions `CompOpt.pSgd_init` and `CompOpt.proxlin_init`, respectively. They accept a problem instance, the number
of desired iterations and a parameter `delta` which produces an initial
estimate `delta`-close to the ground truth (in normalized distance).
The functions implementing the prox-linear method accept extra arguments
controlling the way proximal subproblems are solved, and therefore vary
slightly across problem types, since we provide theoretical guarantees for modified versions of
the prox-linear method for matrix completion and robust PCA. The user is
invited to look at the documentation for a more complete overview.

See an example below, which applies the prox-linear method to a matrix
completion problem starting from an estimate with normalized distance `1.5`
from the solution set.

```julia
prob = CompOpt.genMatCompProb(100, 5, 0.25);
Xest, ds = CompOpt.proxlin_init(prob, 15, 1.5)
```

In the above, `Xest` contains the final estimate after 15 iterations, and `ds`
contains the normalized error across all iterates.
