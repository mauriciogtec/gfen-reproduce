# %%
using Pkg; Pkg.activate("GraphFusedElasticNet.jl")
# using Revise
using Distributions
import StatsBase.weights
using Random
# using RCall
using ProgressMeter
using FileIO
using DataFrames, CSV
using Printf
using Base.Threads
using NPZ
using GraphFusedElasticNet

Random.seed!(138590)

println("Running with $(Threads.nthreads())")

debug = true
if debug
    runid = 0
else
    runid =  parse(Int, ENV["SLURM_PROCID"])
end

# %%
function generate_trace(task, N)
    cuts = [N ÷ 3, 2(N ÷ 3)]
#     cuts = sort(sample(2:N-1, 2, replace=false))
    x1 = 1:cuts[1]
    x2 = (cuts[1] + 1):cuts[2]
    x3 = (cuts[2] + 1):N
    x = [x1; x2; x3]
    
    values1 = 2rand(Uniform(), 4) .- 1.0

    if task == "smooth"
        y1 = values1[1] .+ (values1[2] - values1[1]) .* (x1 ./ cuts[1])
        y2 = values1[2] .+ (values1[3] - values1[2]) .* (x2 .- cuts[1]) ./ (cuts[2] - cuts[1])
        y3 = values1[3] .+ (values1[4] - values1[3]) .* (x3 .- cuts[2]) ./ (N - cuts[2])
    elseif task == "constant"
        y1 = fill(values1[1], cuts[1])
        y2 = fill(values1[2], cuts[2] - cuts[1])
        y3 = fill(values1[3], N - cuts[2])
    elseif task == "mixed"
        y1 = values1[1] .+ Int(rand() < 0.5) .* (values1[2] - values1[1]) .* (x1 ./ cuts[1])
        y2 = values1[2] .+ Int(rand() < 0.5) .* (values1[3] - values1[2]) .* (x2 .- cuts[1]) ./ (cuts[2] - cuts[1])
        y3 = values1[3] .+ Int(rand() < 0.5) .* (values1[4] - values1[3]) .* (x3 .- cuts[2]) ./ (N - cuts[2])
    else
        throw(ArgumentError)
    end
    μ = [y1; y2; y3]
    return μ
end


function generate_spt_task(task_space, task_time, N, pmiss; σ=0.3, outliers=false, type="mixed")
    μ1s = 2generate_trace(task_space, N)
    μ2s = 2generate_trace(task_space, N)
    ts1 = 1.5generate_trace(task_time, N)
    ts2 = 1.5generate_trace(task_time, N)
    
    min_x = -2.5
    max_x = 2.5
    evalpts = collect(range(min_x, max_x, length=100))
    
    if type == "add"
        μs = [(t1 + μ1, t2 + μ2) for (μ1, μ2) in zip(μ1s, μ2s), (t1, t2) in zip(ts1, ts2)]
    elseif type == "mixed"
        μs = [(t1 * μ1, t2 * μ2) for (μ1, μ2) in zip(μ1s, μ2s), (t1, t2) in zip(ts1, ts2)]
    else
        throw(Exception)
    end
    
    dmodels = [
        begin
            dist1 = Truncated(Normal(μ1, σ), min_x, max_x)
            dist2 = Truncated(Normal(μ2, σ), min_x, max_x)
            MixtureModel([dist1, dist2], [0.5, 0.5])
        end
        for (μ1, μ2) in μs
    ]

    devals = [pdf.(d, evalpts) for d in dmodels];
    ndata = [sample([0, 10], weights([pmiss, 1.0 - pmiss])) for d in dmodels]
    y = [rand(d, n) for (d, n) in zip(dmodels, ndata)]
    
    if outliers
        Nobs = sum([1 for n in ndata if n > 0])
        K = Int(floor(Nobs * 0.5))
        idx = sample([i for (i, n) in enumerate(ndata) if n > 0], K, replace=false)
        for i in idx
            j = rand(1:length(y[i]))
            y[i][j] += rand([-1, 1]) * 5.0
        end
    end
                    
    # make matrix pts
    xrange = collect(1:N)
    # temporal
    ptr = Int[]
    brks = Int[1]
    for i in 1:N
        append!(ptr, xrange .+ (i - 1) * N)
        push!(brks, brks[end] + N)
    end
    istemporal = fill(true, N^2)
    # spatial
    xrange = [(i - 1) * N + 1 for i in 1:N]
    for i in 1:N
        append!(ptr, xrange .+ (i - 1))
        push!(brks, brks[end] + N)
    end
    append!(istemporal, fill(false, N^2))
    
    return Dict("evalpts" => evalpts,
                "dmodels" => dmodels,
                "devals" => devals,
                "y" => y,
                "ndata" => ndata,
                "mean1" => μ1s,
                "mean2" => μ2s,
                "t" => ts1,
                "t" => ts2,
                "means" => μs,
                "ptr" => ptr,
                "brks" => brks,
                "istemporal" => istemporal)
end

# function for cross-validation fit

function generate_cvsets(y, nsplits)
    # make the cv splits
    N = length(y)
    cvsets = [Set{Int}() for i in 1:nsplits]
    iobs = shuffle([i for (i, yi) in enumerate(y) if !isempty(yi)])
    Nobs = length(iobs)
    splitsize = Nobs ÷ nsplits
    for k in 1:nsplits
        for i in ((k - 1) * splitsize + 1):(k * splitsize)
            push!(cvsets[k], iobs[i])
        end
    end
    return cvsets
end



function fit2(ytrain, ptr, brks, λ1, λ2, η1, η2, istemporal; parallel=false)
    # create the tree
    depth = 5
    nsplits = 2^depth - 1
    splits = uniform_binary_splits(-2.5, 2.5, depth)

    modelopts = Dict{Symbol, Any}(
        :admm_balance_every => 20,
        :admm_init_penalty => 10.0,
        :admm_residual_balancing => true,
        :admm_adaptive_inflation => false,
        :reltol => 1e-2,
        :admm_min_penalty => 0.1,
        :admm_max_penalty => 32.0,
        :abstol => 1e-6,
        :save_norms => false,
        :save_loss => false
    )
    fitopts = Dict{Symbol, Any}(
        :walltime => 60.0,
        :parallel => parallel
    )
   
    # fit binomial model in each split
    num_nodes = length(ytrain)
    lambdasl1 = Float64[temp ? η1 : λ1 for temp in istemporal]
    lambdasl2 = Float64[temp ? η2 : λ2 for temp in istemporal]
    betas = zeros(nsplits, num_nodes)
    for j in 1:nsplits
        l, m, u = splits.lows[j], splits.mids[j], splits.ups[j]
        successes = [sum(l .< yᵢ .<= m) for yᵢ in ytrain] .+ 1e-8
        attempts = [sum(l .< yᵢ .<= u) for yᵢ in ytrain] .+ 2e-8
        model = BinomialGFEN(ptr, brks, lambdasl1, lambdasl2; modelopts...)
        fit!(model, successes, attempts; fitopts...)
        betas[j, :] = model.beta
    end
    splits, betas
end

                
function cv_fit2(y, evalpts, ptr, brks, istemporal, lambdas, cvsets, models)
    # override only one cvsplit
    nsplits = 1

    # for each cv split get the mse error
    num_nodes = length(y)
    nsplits = length(cvsets)
    nlambdas = length(lambdas)
    test_nloglikelihood = zeros(nlambdas, nsplits)
                    
    # prepare the tree structure and the bint 
    for k in 1:nlambdas
        λ1, λ2, η1, η2 = lambdas[k]
        for i in 1:nsplits
            # get the cv vector with missing data
            ytrain = [(j ∈ cvsets[i] ? Float64[] : yi) for (j, yi) in enumerate(y)]

            splits, betas = fit2(ytrain, ptr, brks, λ1, λ2, η1, η2, istemporal)         
            lows, mids, ups = splits.lows, splits.mids, splits.ups

            # compute the out-of-sample likelihood
            Ntest = 0
            nloglikelihood = 0.0
            for j in cvsets[i]
                root = make_tree_from_bfs(lows, mids, ups, betas[:, j])
                test_eval = y[j]
                Ntest += length(test_eval)
                for yᵢ in test_eval
                    nloglikelihood += - eval_logdens(root, yᵢ)
                end
            end
            nloglikelihood /= Ntest
            test_nloglikelihood[k, i] = nloglikelihood
        end
    end
    # println(".")

    # now choose the best lambdas
    test_nloglikelihood = dropdims(mean(test_nloglikelihood, dims=2), dims=2)
    best_lambdas = lambdas[argmin(test_nloglikelihood)]
    best_nloglikelihood = minimum(test_nloglikelihood)
    
    # compute overall likelihood
    nsims = 100
    samples = [rand(model, nsims) for model in models]
    λ1, λ2, η1, η2 = best_lambdas
    splits, betas = fit2(y, ptr, brks, λ1, λ2, η1, η2, istemporal, parallel=true)
    lows, mids, ups = splits.lows, splits.mids, splits.ups

    lls = zeros(num_nodes)
    for i in 1:num_nodes
        beta = betas[:, i]
        root = make_tree_from_bfs(lows, mids, ups, beta)
        ll = mean([-eval_logdens(root, yᵢ) for yᵢ in samples[i]])
        lls[i] = ll
    end
    validation_nloglikelihood = mean(lls)

    return Dict("best_lambdas" => best_lambdas,
                "cv_nloglikelihood" => best_nloglikelihood,
                "val_nloglikelihood" => validation_nloglikelihood)
end

function get_hypers(method)
    lambdas_dict = Dict(
        "fl" => [(10^l, 1e-12) for l in range(-2, log10(5.0), length=20)],
        "kal" => [(1e-12, 10^x) for x in range(-2, log10(5.0), length=20)],
        "enet" => [
            (10^l1, 10.0^x) for l1 in range(-2, log10(5.0), length=10) for x in range(-2, log10(5.0), length=10)
        ]
    ) 
    ls = lambdas_dict[method]
    hypers = [(λ1, λ2, η1, η2) for (λ1, λ2) in ls for (η1, η2) in ls]
    # if method == "enet"
    #     M = 400
    # else
    #     M = 100
    # end
    M = 10
    return rand(hypers, M)
end

##

function run_benchmarks(N, pmiss;
                        nsims=100,
                        nsplits=5,
                        tasks=("constant", "smooth", "mixed"),
                        outliers=false)
    experiment_results = []
    for task_space in tasks
        for task_time in tasks
            for method in ("fl", "kal", "enet")
                println("Running task_space $task_space task_time $task_time for method $method, outliers $outliers")
                
                for l in 1:nsims
                    D = generate_spt_task(task_space, task_time, N, pmiss, outliers=outliers)
                    y = vec(D["y"])
                    models = vec(D["dmodels"])
                    ndata = vec(D["ndata"])
                    devals = vec(D["devals"])
                    ptr = D["ptr"]
                    brks = D["brks"]
                    evalpts = D["evalpts"]
                    istemporal = D["istemporal"]
                    
                    cvsets = generate_cvsets(y, nsplits)

                    lambdas = get_hypers(method)
                    results = cv_fit2(y, evalpts, ptr, brks, istemporal, lambdas, cvsets, models)

                    new_result = Dict(
                        :experiment => l,
                        :task_space => task_space,
                        :task_time => task_time,
                        :method => method,
                        :cv_nloglikelihood => results["cv_nloglikelihood"],
                        :val_nloglikelihood => results["val_nloglikelihood"])
                    push!(experiment_results, new_result)
                end
            end
        end
    end
    return experiment_results
end

# %%
# see examples
task = generate_spt_task("mixed", "mixed", 30, 0.8, outliers=false)
evalpts = task["evalpts"]
neval = length(evalpts)
devals = zeros(30, 30, neval)
for j in 1:30, i in 1:30
    devals[i, j, :] = task["devals"][i, j]
end
# fit on sample data tor example

λ1, λ2, η1, η2 = 0.25, 0.05, 0.25, 0.05
ptr, brks, istemporal = task["ptr"], task["brks"], task["istemporal"]
y = reshape(task["y"], :)
splits, betas = fit2(y, ptr, brks, λ1, λ2, η1, η2, istemporal)
betas = reshape(betas, :, 30, 30)
splitvals = splits.splitvals
fitprobs = zeros(30, 30, length(splitvals))
fitdens = zeros(30, 30, length(splitvals))
nsplits = size(betas, 1)
δx = diff(splitvals)
δx = [δx[1]; δx]
for i in 1:30, j in 1:30
    root = make_tree_from_bfs(splits.lows, splits.mids, splits.ups, betas[:, i, j])
    fitprobs[i, j, :] = [exp(eval_logprob(root, x)) for x in splitvals]
    fitdens[i, j, :] = [exp(eval_logdens(root, x)) for x in splitvals]
end

tosave = Dict(
    "evalpts" => evalpts,
    "devals" => devals,
    "ndata" => task["ndata"],
    "splitvals" => splitvals,
    "fitprobs" => fitprobs,
    "fitdens" => fitdens,
)
# npzwrite("simulations/example_simulation_task.npz", tosave)


##
# %%
N = 30
pmiss = 0.8
nsims = 1
tasks = ("smooth", "constant", "mixed")

experiment_results = run_benchmarks(N, pmiss, nsims=nsims, tasks=tasks)
experiment_results_out = run_benchmarks(N, pmiss, nsims=nsims, tasks=("mixed", ), outliers=true)


# %%
df = DataFrame(experiment = Int[],
               task_space=String[],
               task_time=String[],
               method=String[],
               cv_nloglikelihood=Float64[],
               cv_nloglikelihood=Float64[])
for record in [experiment_results; experiment_results_out]
    push!(df, record)
end
df.runid = fill(runid, size(df, 1))

fname = @sprintf("simulations/benchmarks/results-spt-%03d.csv", runid)
CSV.write(fname, df)
