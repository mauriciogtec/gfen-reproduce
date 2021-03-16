using Pkg; Pkg.activate("GraphFusedElasticnet.jl/")
include("GraphFusedElasticNet.jl/src/ARS.jl")
using Plots


μ, σ = 1.0, 2.0
f(x) = exp(-0.5(x - μ)^2 / σ^2) / sqrt(2pi * σ^2) 
support = (-10.0, 2.0)

# Build the sampler and simulate 10,000 samples
sampler = RejectionSampler(f, support, (-1.0, 1.0), max_segments = 3)
@time sim = run_sampler!(sampler, 1000);

x = range(-10.0, 10.0, length=100)
envelop = [eval_envelop(sampler.envelop, xi) for xi in x]
target = [f(xi) for xi in x]

histogram(sim, normalize = true, label = "Histogram")
plot!(x, [target envelop], width = 2, label = ["Normal(μ, σ)" "Envelop"])