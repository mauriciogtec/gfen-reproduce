using Pkg
Pkg.activate("./GraphFusedElasticNet.jl/")
using DelimitedFiles
using CSV
using DataFrames
using Dates
using DataStructures
using Printf
using ProgressMeter
using Statistics
using Base.Threads
using NPZ
using GraphFusedElasticNet
# using RCall  # for plots
using Plots

##
eval_samples_at_examples_only = true


# tazs of interest
example_ids = [
    "Airport",
    "The Domain",
    "Pflugerville",
    "University",
    "Downtown",
    "Red & 12th"
]
example_tazs = [499, 201, 160, 362, 1951, 1898]

# key, val = zip(collect(siteids)...)

##
splits = readtable("processed_data/splits_qua.csv")
vertexinfo = readtable("processed_data/vertex_data.csv")
num_nodes = size(vertexinfo, 1)
num_splits = size(splits, 1)


# overrride to evaluate everywhere
num_sites = length(unique(vertexinfo.taz))
num_times = length(unique(vertexinfo.hour))
tazs = unique(vertexinfo.taz)
tazmap = Dict(x => i for (i, x) in enumerate(tazs))
vertexmap = Dict((r.taz, r.hour) => r.vertex + 1 for r in eachrow(vertexinfo))

##
betafiles = readdir("best_betas")
betafiles_bayesian = readdir("best_betas_bayesian")

##

x = npzread("best_betas_bayesian/$(betafiles_bayesian[1])")
num_samples = size(x, 2)
tazs_samples = (
    eval_samples_at_examples_only ? example_tazs : tazs
)
num_sites_samples = length(tazs_samples)

##

maps = zeros(Float32, num_sites, num_times, num_splits)
samples = zeros(Float32, num_sites_samples, num_times, num_samples, num_splits)
# samples = samples[:, :, 1:50, :]

println("Extracting data from saved estimates...")
pbar = Progress(num_splits)
@threads for j in 1:num_splits
    f = betafiles_bayesian[j]
    g = betafiles[j]
    x = npzread("best_betas_bayesian/$f")
    y = readdlm("best_betas/$g")
    for i in 1:num_sites_samples
        for t in 1:num_times
            v = vertexmap[tazs_samples[i], t]
            samples[i, t, :, j] = x[v, :]
        end
    end
    for i in 1:num_sites
        for t in 1:num_times
            v = vertexmap[tazs[i], t]
            maps[i, t, j] = y[v]
        end
    end
    next!(pbar)           
end

##

mid = Float32.(splits.mid)
lows = Float32.(splits.lows)
ups = Float32.(splits.ups)

##
xseq = Float32[minimum(splits.lows); sort(splits.mid); maximum(splits.ups)]
# x2 = 0.5f0 * (xseq[1:end - 1] + xseq[2:end])
# x3 = 0.5f0 * (x2[1:end - 1] + x2[2:end])
# xseq = sort([xseq; x2; x3])
N = length(xseq)
support = (lows[1], ups[1])
log_probs = fill(Float32(-Inf), (num_sites_samples, num_times, num_samples, N))
log_probs_map = fill(Float32(-Inf), (num_sites, num_times, N))


##
println("Computing densities from MAP estimates")
pbar = Progress(num_sites)
@threads for i in 1:num_sites
    for t in 1:num_times
        betas = maps[i, t, :]
        root = make_tree(lows, mid, ups, betas)
        for j in 1:N
            if support[1] ≤ xseq[j] ≤ support[2]
                log_probs_map[i, t, j] = eval_logprob(root, xseq[j])
            end
        end  
    end
    next!(pbar)
end

##
println("Computing densities from Bayesian samples")
pbar = Progress(num_sites_samples)
@threads for i in 1:num_sites_samples
    for t in 1:num_times
        for s in 1:num_samples
            betas = samples[i, t, s, :]
            root = make_tree(lows, mid, ups, betas)
            for j in 1:N
                if support[1] ≤ xseq[j] ≤ support[2]
                    log_probs[i, t, s, j] = eval_logprob(root, xseq[j])
                end
            end
        end
    end
    next!(pbar)
end # not necessary anymore

pmat = exp.(log_probs)
log_probs = nothing
pmat ./= sum(pmat, dims=4)

#

pmat_map = exp.(log_probs_map)
pmat_map ./= sum(pmat_map, dims=3)

npzwrite("fitted_densities/map.npy", pmat_map)

# pmat_median = mapslices(median, pmat, dims=3)
# pmat_median = dropdims(pmat_median, dims=3)
# pmat_median ./= sum(pmat_median, dims=3)

# pmat_q25 = mapslices(x -> quantile(x, 0.25), pmat, dims=3)
# pmat_q25 = dropdims(pmat_q25, dims=3)
# pmat_q25 ./= sum(pmat_q25, dims=3)

# pmat_q75 = mapslices(x -> quantile(x, 0.75), pmat, dims=3)
# pmat_q75 = dropdims(pmat_q75, dims=3)
# pmat_q75 ./= sum(pmat_q75, dims=3)

# pmat_q05 = mapslices(x -> quantile(x, 0.05), pmat, dims=3)
# pmat_q05 = dropdims(pmat_q05, dims=3)
# pmat_q05 ./= sum(pmat_q05, dims=3)

# pmat_q95 = mapslices(x -> quantile(x, 0.95), pmat, dims=3)
# pmat_q95 = dropdims(pmat_q95, dims=3)
# pmat_q95 ./= sum(pmat_q95, dims=3)

# pmat_iqr = pmat_q75 - pmat_q25

# pmat_mean = mapslices(mean, pmat, dims=3)
# pmat_mean = dropdims(pmat_mean, dims=3)
# pmat_mean ./= sum(pmat_mean, dims=3)

# means = sum(reshape(xseq, 1, 1, 1, :) .* pmat, dims=4)
# means = dropdims(means, dims=4)

# ##
# minimum(means)
# maximum(means)

##

# nid = 5
# t = 124
# taz = example_tazs[nid]
# posts = pmat[nid, t, :, :]'
# taznum = tazmap[taz]
# # p = plot(xseq, posts, c=:gray, alpha=0.2, label="", lw=2)
# y = pmat_map[taznum, t, :]
# δ = diff(xseq)
# δ = [δ[1]; δ]
# p = plot(xseq, y ./ δ, c=:blue, lw=2, alpha=0.5, lab="map")
# title!("$(example_ids[nid]) at time $t")
# xlabel!("productivity (dollar/hour)")
# ylabel!("density")
# up = (pmat_q95[nid, t, :] - pmat_median[nid, t, :])
# bot = (pmat_median[nid, t, :] - pmat_q05[nid, t, :])
# plot!(
#     p, xseq, pmat_median[nid, t, :];
#     ribbon=[up, bot], c=:red, lw=2, alpha=0.5, lab="median"
# )
# # plot!(p, xseq, pmat_mean[nid, t, :], c=:orange, lw=2, alpha=0.5)
# xlims!(0.0, 100.0)

## where to obtain pointwise estimates from
src = pmat_map
# src = pmat_median

# ##
tail_quantities = [18.56, 21.64, 32.73, 34.74]
pmats_point_tp = []
for tq in tail_quantities
    tprob(u) = sum(ui for (ui, xi) in zip(u, xseq) if xi ≥ tq)
    mat = dropdims(mapslices(tprob, src, dims=3), dims=3)
    push!(pmats_point_tp, mat)
end
tprob(u) = sum(ui for (ui, xi) in zip(u, xseq) if xi ≥ 21.64)
# pmats_tp21 = dropdims(mapslices(tprob, pmat, dims=4), dims=4)

npzwrite("fitted_densities/map_tp18.npy", pmats_point_tp[1])
npzwrite("fitted_densities/map_tp21.npy", pmats_point_tp[2])
npzwrite("fitted_densities/map_tp32.npy", pmats_point_tp[3])
npzwrite("fitted_densities/map_tp34.npy", pmats_point_tp[4])
# npzwrite("fitted_densities/posterior_tp21.npy", Float16.(pmats_tp21))


# Variatiblity in Exceeding living wage?
minimum(pmats_point_tp[1])
maximum(pmats_point_tp[1])


# pmats_point_tp = nothing
# pmats_tp21 = nothing

# ##
bottoms = Float32.([0.1, 0.25, 0.5, 0.75])
pmats_point_bot = []

function dquant(u::AbstractVector{T}, lev::T)::T where {T<:AbstractFloat}
    cprobs = cumsum(u)
    ix = findfirst(u -> (u ≥ lev), cprobs)
    y1 = cprobs[ix]
    x1 = xseq[ix]
    if ix > 0
        y0 = cprobs[ix - 1]
        x0 = xseq[ix - 1]
        m = (y1 - y0) / (x1 - x0)  # y = m * (x - x0) + y0
        return (lev - y0) / m + x0  # interpolate quantile
    else
        return xseq[1]
    end
end

for lev in bottoms
    mat = dropdims(mapslices(u -> dquant(u, lev), src, dims=3), dims=3)
    push!(pmats_point_bot, mat)
end

# pmats_bot10 = dropdims(mapslices(u -> dquant(u, 0.1f0), pmat, dims=4), dims=4)
# histogram(vec(pmats_point_bot[1]))
mean(pmats_point_bot[1] .< 10.0)


## map estimates

npzwrite("fitted_densities/map_q10.npy", pmats_point_bot[1])
npzwrite("fitted_densities/map_q25.npy", pmats_point_bot[2])
npzwrite("fitted_densities/map_q50.npy", pmats_point_bot[3])
npzwrite("fitted_densities/map_q75.npy", pmats_point_bot[4])
# npzwrite("fitted_densities/posterior_q10.npy", Float16.(pmats_bot10))



## bayesian
if eval_samples_at_examples_only
    pmat_examples = pmat
else
    pmat_examples = pmat[[tazmap[taz] for taz in example_tazs], :, :, :]
end
# note: saving in logits reduces loss of precision in float16
npzwrite("fitted_densities/examples_posterior_density_logits.npy", Float16.(log.(pmat_examples)))


## additional info
open("fitted_densities/evaluation_points.csv", "w") do io
    writedlm(io, xseq)
end


##
example_info = DataFrame(
    :name => example_ids,
    :taz => example_tazs,
    :row => [tazmap[i] for i in example_tazs]
)
CSV.write("fitted_densities/examples_info.csv", example_info)
