# %% 
using Pkg
Pkg.activate("./GraphFusedElasticNet.jl/")
Pkg.instantiate()
using GraphFusedElasticNet
using LightGraphs
using DelimitedFiles
using JSON
using DataStructures


# %% 
fname ="./processed_data/spatiotemporal_graph.csv"
edgeinfo = readdlm(fname, ',', Int, skipstart=1)
edgeinfo[1:10,:]

# %% use our utils from GFEN library for trail extraction
graph = graph_from_edgelist(edgeinfo[:,1:2], from_zero=true)
trails = find_trails(graph, ntrails=64)


# %% identify temporal edges
# create temporal edge dict
tmpdict = DefaultDict{Tuple{Int, Int}, Bool}(false)
cols = [edgeinfo[:,i] for i in 1:3]
for (k, (i, j, t)) in enumerate(zip(cols...))
    x, y = min(i, j), max(i, j)
    tmpdict[x, y] = t
end
# now tag along ptr
ptr = trails.ptr
istemporal = zeros(Int, length(ptr))
brks_ = OrderedSet(trails.brks)
for (k, (i, j)) in enumerate(zip(ptr[1:end-1], ptr[2:end]))
    if k + 1 ∉ brks_
        x, y = min(i, j), max(i, j)
        istemporal[k] = tmpdict[x, y]
    end
end

# %% save fields as json
data = Dict("ptr" => trails.ptr,
            "brks" => trails.brks,
            "wts" => trails.wts,
            "istemp" => istemporal,
            "num_nodes" => trails.num_nodes)
open("./processed_data/trails.json","w") do f
    JSON.print(f, data)
end