using Pkg
Pkg.activate("./GraphFusedElasticNet.jl/")
Pkg.instantiate()
using DelimitedFiles
using CSV
using DataFrames
using Dates
using DataStructures
using Printf
using ProgressMeter
using Statistics
using Base.Threads

##

##
betafiles = readdir("best_betas")

##
splits = readtable("processed_data/splits_opt_pt.csv")

##
vertexinfo = readtable("processed_data/vertex_data.csv")
num_nodes = size(vertexinfo, 1)


# output settings
# optionally we can set a numer of files for the output
# so that file size is smaller, in this case, they can best_betas
# read in order and reassambled by horizontal concatenation
num_files = 20
smoothprobs_relpath() = "output_smooth_probs/"

##

function getsplit(s::String)
    x = split(s, '_')[3]
    x = split(x, '.')[1]
    return x
end

##

# read files with betas
num_splits = size(splits, 1)
betas = zeros(num_nodes, num_splits)
for (j, file) in enumerate(betafiles)
    j = parse(Int, file[7:8]) + 1
    betas[:, j] = vec(readdlm("best_betas/$file"))
end

##

mid = Float64.(splits.mid)
lows = Float64.(splits.lows)
ups = Float64.(splits.ups)
root = make_tree_from_bfs(lows, mid, ups)


# nodes of interest



#


N = 100
xseq = collect(range(0.0, 125.0, length=N))
support = (lows[1], ups[1])
log_probs = fill(-Inf, (num_nodes, N))
pbar = Progress(num_nodes)

@threads for i in 1:num_nodes
    root = make_tree_from_bfs(lows, mid, ups, betas[i, :])
    for j in 1:N
        if support[1] ≤ xseq[j] ≤ support[2]
            log_probs[i, j] = eval_logprob(root, xseq[j])
        end
    end
    next!(pbar)
end


##

pmat = exp.(log_probs)
pmat ./= sum(pmat, dims=2)

##

pmatt = pmat'[:, range(1, num_nodes, step=100)]
plot(pmatt, c=:gray, alpha=0.05)

##

##
means = vec(sum(pmat .* xseq', dims=2))
histogram(means)