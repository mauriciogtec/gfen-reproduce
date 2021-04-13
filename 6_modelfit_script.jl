using Pkg
Pkg.activate("GraphFusedElasticNet.jl")
# Pkg.instantiate()

using DataFrames
using Printf
using DelimitedFiles
using JSON
using NPZ
using GraphFusedElasticNet


#
debug = false
if debug
    split_to_fit = 0
else
    split_to_fit =  parse(Int, ENV["SLURM_PROCID"])
end

##
splits = readtable("processed_data/splits_qua.csv")
split = splits[split_to_fit + 1, :]

##
fname = @sprintf("productivity_splits/%02d.csv", split_to_fit)
splitdata = readdlm(fname, ',')
println("Fitting to data $(fname)...")

## find be[s]t lambda
fname = @sprintf("modelfit_metrics/cvloss_%02d.csv", split_to_fit)
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
ϵ = 1e-3
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
    :admm_balance_every => 20,
    :admm_init_penalty => 1.0,
    :admm_residual_balancing => true,
    :admm_adaptive_inflation => true,
    :reltol => 1e-4,
    :admm_min_penalty => 0.25,
    :admm_max_penalty => 32.0,
    :abstol => 1e-7,
    :save_norms => false,
    :save_loss => false
)
fitopts = Dict{Symbol, Any}(
    :walltime => 3600.0,
    :parallel => true
)


## define and train model
fname = @sprintf("../gfen-reproduce/best_betas/betas_%02d.csv", split_to_fit)


println("Fitting MAP model...")
map_mod = BinomialGFEN(ptr, brks, lambdasl1, lambdasl2; modelopts...)
@time converged = fit!(map_mod, s, a ; fitopts...)
println("MAP estimate converged: $(converged)")
println("Saving MAP estimate to $fname... in Float16")
open(fname, "w") do io
    writedlm(io, Float16.(map_mod.beta), ',')
end

##

println("Loading MAP model...")
mcmc_init = vec(readdlm(fname, ','))

##

edges = [(r.vertex1 + 1, r.vertex2 + 1) for r in eachrow(edges_df)]
tv1 = [(r.temporal == 1) ? λtl1 : λsl1 for r in eachrow(edges_df)]
tv2 = [(r.temporal == 1) ? λtl2 : λsl2 for r in eachrow(edges_df)]

println("Initializing MCMC chain...")
bmod = BayesianBinomialGFEN(edges, tv1=tv1, tv2=tv2)


## fit model
# runs took anytime between 3 and 7 hours
n = 5500
thinning = 5
burnin = 4500

# n = 200
# thinning = 1
# burnin = 0

#
init = mcmc_init  # zeros(size(mcmc_init))
@time chain = sample_chain(bmod, s, a, n, burnin=burnin, thinning=thinning, init=init, verbose=false, async=true)

##
fname = @sprintf("best_betas_bayesian/%02d.npy", split_to_fit)
println("Saving chain $(fname) in Float16...")


# open(fname, "w") do io
#     writedlm(io, Float16.(chain), ',')
# end
npzwrite(fname, Float16.(chain))
