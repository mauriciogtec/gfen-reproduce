using Pkg
Pkg.activate("./GraphFusedElasticNet.jl/")
Pkg.instantiate()
using DelimitedFiles
using CSV
using DataFrames


workdir() = pwd()
splits_relpath() = "processed_data/splitlevels.csv"
vertexinfo_relpath() = "processed_data/vertex_data.csv"
betas_relpath() = "best_betas/"
betafiles = readdir(joinpath(workdir(), betas_relpath()))

# output settings
# optionally we can set a numer of files for the output
# so that file size is smaller, in this case, they can best_betas
# read in order and reassambled by horizontal concatenation
num_files = 20
smoothprobs_relpath() = "output_smooth_probs/"


# read the data from the graph
# actualy we only need the number of nodes
types=Dict(
    "nodes" => String,
    "vertex" => Int,
    "taz" => Int,
    "hours" => Int)
graphdata = CSV.read(vertexinfo_relpath(), types=types)
num_nodes = size(graphdata, 1)
               

function getsplit(s::String)
    x = split(s, '_')[3]
    x = split(x, '.')[1]
    return x
end

# read files with betas
betas = Dict{Tuple{Int, Int}, Vector{Float64}}()
for file in betafiles
    bin = getsplit(file)
    a, b = parse.(Int, String.(split(bin, '-')))
    β = vec(readdlm(joinpath(betas_relpath(), file)))
    betas[a, b] = β
end

splitlevels = vec(readdlm(splits_relpath()))
nbins = length(splitlevels) - 1

# descend in each bin and multiplyprobabilities
probs = ones(nbins, num_nodes)
for (key, value) in betas
    a = key[1] + 1
    b = key[2]
    m = (a + b) ÷ 2
    p = 1.0 ./ (1.0 .+ exp.(-value))
    p = reshape(p, 1, :)
    probs[a:m, :] .*= p
    probs[m:b, :] .*= 1.0 .- p
end
probs ./= sum(probs, dims=1)

# write output
if num_files == 1
    fname = joinpath(smoothprobs_relpath(), "probs.csv")
    writedlm(fname, probs, ',')
else
    chunksize = if num_nodes % num_files == 0
        num_nodes ÷ num_files
    else
        (num_nodes ÷ num_files) + 1
    end
    for i in 1:num_files
        from = (i - 1) * chunksize + 1
        to = min(i * chunksize, num_nodes)
        chunk = probs[:, from:to]
        fname = joinpath(smoothprobs_relpath(), "probs-chunk_$i.csv")
        writedlm(fname, chunk, ',')
    end
end


# probs to densities
# Δ = splitlevels[2:end] .- splitlevels[1:end-1]
# dens = probs ./ Δ
