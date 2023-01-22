`Documenter.jl` Compatible Commonmark

================
pat-alt
1/22/23

## Introduction

This Quarto notebook generates Markdown output (`.md`) that works nicely with `Documenter.jl`.

## Math

Let $t\in\{0,1\}$ denote the target label, $M$ the model (classifier) and x′ ∈ ℝᴰ the vector of counterfactual features. In order to generate recourse the `GenericGenerator` optimizes the following objective function through steepest descent

``` math
x\prime = \arg \min_{x\prime}  \ell(M(x\prime),t) + \lambda h(x\prime)
```

where $\ell$ denotes some loss function targeting the deviation between the target label and the predicted label and $h(\cdot)$ is a complexity penalty generally addressing the *realism* or *cost* of the proposed counterfactual.

Let’s generate some toy data:

``` julia
# Some random data:
using CounterfactualExplanations.Data
Random.seed!(1234)
N = 25
w = [1.01 1.0]# true coefficients
b = 0
xs, ys = Data.toy_data_linear(N)
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')
plt = plot()
plt = scatter!(counterfactual_data)
savefig(plt, joinpath(www_path, "binary_samples.png"))
```
