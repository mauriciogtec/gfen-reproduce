using Pkg; Pkg.activate("GraphFusedElasticNet.jl")
using DataFrames
using Printf
using DelimitedFiles
using JSON
using GraphFusedElasticNet
using NPZ

debug = false
if debug
    split_to_fit = 3
else
    split_to_fit = (length(ARGS) > 0 && isa(ARGS[1], Int)) ? (ARGS[1] + 1) : 3
end
##
splits = readtable("processed_data/splits_opt_pt.csv")
split = splits[split_to_fit, :]

##
fname = @sprintf("productivity_splits/%02d.csv", split_to_fit + 1)
splitdata = readdlm(fname, ',')
println("Fitting to data $(fname)...")

## find best lambda
fname = @sprintf("modelfit_metrics/cvloss_%02d.csv", split_to_fit + 1)
best_lams = readtable(fname)
best_row = argmax(best_lams.final_pred)
λsl1 = best_lams.λsl1[best_row]
λsl2 = best_lams.λsl2[best_row]
λtl1 = best_lams.λtl1[best_row]
λtl2 = best_lams.λtl2[best_row]

##
edges_df = readtable("../gfen-reproduce/processed_data/spatiotemporal_graph.csv")

##
num_nodes = size(splitdata, 1)
num_edges = size(edges_df, 1)

## load data and map estimate for fast mixing
fname = @sprintf("../gfen-reproduce/best_betas/betas_%02d.csv", split_to_fit)
init = vec(readdlm(fname, ','))
ϵ = 1e-6
s = splitdata[:, 1] .+ ϵ
a = splitdata[:, 2] .+ 2ϵ

## fit MAP model
trails = JSON.parsefile("processed_data/trails.json")
istemp = Bool.(trails["istemp"])
wts = Float64.(trails["wts"])
ptr = Int.(trails["ptr"])
brks = Int.(trails["brks"])
lambdasl1 = [w * (t ? λtl1 : λsl1) for (t, w) in zip(istemp, wts)]
lambdasl2 = [w * (t ? λtl2 : λsl2) for (t, w) in zip(istemp, wts)]

##
modelopts = Dict{Symbol, Any}(
    :admm_balance_every => 10,
    :admm_init_penalty => 0.1,
    :admm_residual_balancing => true,
    :admm_adaptive_inflation => true,
    :reltol => 1e-3,
    :admm_min_penalty => 0.1,
    :admm_max_penalty => 5.0,
    :abstol => 0.,
    :save_norms => false,
    :save_loss => false
)
fitopts = Dict{Symbol, Any}(
    :walltime => 10.0,
    :parallel => true
)


## define and train model
println("Fitting fast MAP mode...")
map_mod = BinomialGFEN(ptr, brks, lambdasl1, lambdasl2; modelopts...)
fit!(map_mod, s, a; fitopts...)

##
edges = [(r.vertex1 + 1, r.vertex2 + 1) for r in eachrow(edges_df)]
tv1 = [(r.temporal == 1) ? λtl1 : λsl1 for r in eachrow(edges_df)]
tv2 = [(r.temporal == 1) ? λtl2 : λsl2 for r in eachrow(edges_df)]

println("Initializing MCMC chain...")
mod = BayesianBinomialGFEN(edges, tv1=tv1, tv2=tv2)
init = map_mod.beta

## fit model
n = 1_000
thinning = 5
burnin = 0.5

##
chain = sample_chain(mod, s, a, n, init=init, verbose=false, async=true)

##
nstart = ceil(Int, size(chain, 2) * burnin)
chain = chain[:, nstart:end]

##
fname = @sprintf("best_betas_bayesian/%02d.npz", split_to_fit)
println("Saving to $(fname) in Float16...")
npzwrite(fname, Float16.(chain))
