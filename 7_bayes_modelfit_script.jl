# set-up distributed environment
using Pkg

# first make sure all packages necessary are loaded
workdir() = pwd()
GFEN_relpath() = "GraphFusedElasticNet.jl/"
Pkg.activate(joinpath(workdir(), GFEN_relpath()))
Pkg.instantiate()


using Distributed


# relevant path (ADAPT TO CLUSTER IF NECESSARY)
@everywhere begin
    workdir() = pwd()
    GFEN_relpath() = "GraphFusedElasticNet.jl/"
    traildata_relpath() = "processed_data/trails.json"
    betas_relpath() = "best_betas/"
    fitmetrics_relpath() = "modelfit_metrics/"
    data_relpath() = "productivity_splits/"
    flist = readdir(joinpath(workdir(), data_relpath()))
end


# only set to true if running interactive
# not if running from batch file
@everywhere RUNNING_INTERACTIVE = false 
if RUNNING_INTERACTIVE
    walltime = 120.0
    ngens = 4
    gensize = 16
    nsplits = 5
    data_targets = flist
else
    walltime = parse(Float64, ARGS[1])
    ngens = parse(Int, ARGS[2])
    gensize = parse(Int, ARGS[3])
    nsplits = parse(Int, ARGS[4])
    data_targets = ARGS[5:end]
end


# instatiate and load libraries in every process
@everywhere begin
    using Pkg
    Pkg.activate(joinpath(workdir(), GFEN_relpath()))
    using JSON
    using DataFrames
    using Dates
    using Distributions
    using Random
    using CSV
    using Base.Threads
    using DelimitedFiles
    using GraphFusedElasticNet
end


# if in slurm cluster environment (TACC)
using ClusterManagers
if haskey(ENV, "SLURM_NTASKS")
    np = parse(Int, ENV["SLURM_NTASKS"])
    addprocs(SlurmManager(np))
end


# data reader
@everywhere function loadtrails()
    fname = joinpath(workdir(), traildata_relpath())
    trails = JSON.parsefile(fname)
    ptr = convert(Vector{Int}, trails["ptr"])
    brks = convert(Vector{Int}, trails["brks"])
    wts = convert(Vector{Float64}, trails["wts"])
    istemp = convert(Vector{Bool}, trails["istemp"])
    num_nodes = Int(trails["num_nodes"])
    return ptr, brks, wts, istemp, num_nodes
end


# generates sets of nodes to take out during cv
@everywhere function load_map_betas(y, nsplits)
    # make the cv splits
    # the only trick is not to count the integers that 
    # have missing data
    
end


# prepare training and test data fro a cvset
@everywhere function get_train_data(y, test_idx)
    train = copy(y)
    for i in 1:size(y, 1)
        if i ∈ test_idx
            train[i, :] = [0.0, 0.0]
        end
    end
    train
end


# useful to compute loglikelihod
@everywhere function sigmoid(x::Float64)::Float64
    1.0 / (1.0 + exp(-x))
end


# fits a binomial GFEN model for given params and data
@everywhere function fit_model(
        ytrain, ptr, brks, wts, istemp,
        λ1, λ2, η1, η2, modelopts, fitopts)
    N = size(ytrain, 1)
    lambdasl1 = [w * (t ? η1 : λ1) for (t, w) in zip(istemp, wts)]
    lambdasl2 = [w * (t ? η2 : λ2) for (t, w) in zip(istemp, wts)]
    # define and train model
    model = BinomialGFEN(ptr, brks, lambdasl1, lambdasl2; modelopts...)
    succ = ytrain[:, 1] .+ 0.05
    att = ytrain[:, 2] .+ 0.1
    fit!(model, succ, att; fitopts...)
    return model
end


# main program
@everywhere function fit_split(filename, walltime, nsplits, ngens, gensize)
    # read trail data (it is then copied to each worker by pmap)
    ptr, brks, wts, istemp, num_nodes = loadtrails()

    # model parameters
    modelopts = Dict{Symbol, Any}(
        :admm_balance_every => 10,
        :admm_init_penalty => 0.1,
        :admm_residual_balancing => true,
        :admm_adaptive_inflation => true,
        :reltol => 1e-3,
        :admm_min_penalty => 0.1,
        :admm_max_penalty => 5.0,
        :abstol => 0.,
        :save_norms => true,
        :save_loss => true)
    fitopts = Dict{Symbol, Any}(
        :walltime => walltime,
        :parallel => false)

    # read data
    fname = joinpath(workdir(), data_relpath(), filename)
    y = readdlm(fname, ',') 
    cvsets = generate_cvsets(y, nsplits)  
    
    # df = Vector{DataFrame}(undef, ngens)
    results = []
    for gen in 1:ngens
        # sample lambdas from generation
        if gen == 1
            if gensize == 16
                # evaluate first in all the corners !
                idx = Int[]
                a, b = minimum(sl1), minimum(sl2)
                A, B = maximum(sl2), maximum(tl2)
                c, d = minimum(tl1), minimum(tl2)
                C, D = maximum(tl1), maximum(tl2)
                for j in 1:nl
                    if (hparams[1, j] in [a, A] &&
                        hparams[2, j] in [b, B] &&
                        hparams[3, j] in [c, C] &&
                        hparams[4, j] in [d, D])
                        push!(idx, j)
                    end
                end
            else
                idx = sample(1:nl, gensize, replace=false)
            end
            prev_pred = fill(-Inf, gensize)
            prev_sd = fill(0.0, gensize)
        else
            idx, _, _, prev_pred, prev_sd  = gpsample(gpsampler, gensize)
        end
        pars = [hparams[:, i] for i in idx]
        slambdasl1 = hparams[1, idx]
        slambdasl2 = hparams[2, idx]
        tlambdasl1 = hparams[3, idx]
        tlambdasl2 = hparams[4, idx]

        # run for every lambda in thread
        fns = fill(filename, gensize)
        pids = fill(getpid(), gensize)
        hosts = fill(gethostname()[1:8], gensize)
        gens = fill(gen, gensize)

        # obtain cv losses and update gaussian process
        usethreads = true
        cv_logll, thrds, avtime, avalpha = cv_eval(
            y, ptr, brks, wts, istemp,
            pars, cvsets,
            modelopts, fitopts, usethreads)
        addobs!(gpsampler, idx, cv_logll)
        # update the offset helps with better estimates
        # at far regions from observed data
        # it's like using an empirical prior, it doesn't 
        # affect much regions with densely observed data
        N0, N1 = (gen - 1) * gensize, gen * gensize
        gp_offset = (gp_offset *  N0 +  sum(cv_logll)) / N1
        gpsampler.offset = gp_offset

        df = DataFrame(
            fn=fns, pid=pids, host=hosts, thread=thrds,
            λsl1=slambdasl1, λsl2=slambdasl2, λtl1=tlambdasl1, λtl2=tlambdasl2,
            cv_logll=cv_logll, time=avtime, alpha=avalpha, gen=gens,
            paramid=idx, prev_pred=prev_pred, prev_sd=prev_sd)
        println(df)
        push!(results, df)
    end

    # now obtain the best hyperperameters and train the best model
    results = vcat(results...)
    pred, band = gpeval(gpsampler)
    results.final_pred = pred[results.paramid]
    results.final_sd = band[results.paramid]
    #
    bestidx = argmax(results.final_pred)
    best_predicted = results.final_pred[bestidx]
    λ1, λ2, η1, η2 = hparams[:, results.paramid[bestidx]]

    gpdf = DataFrame(
        slambdasl1=hparams[1, :],
        slambdasl2=hparams[2, :],
        tlambdasl1=hparams[3, :],
        tlambdasl2=hparams[4, :],
        pred=pred,
        band=band,
        tested=gpsampler.tested)
     
    modelopts[:reltol] /= 10.0
    fitopts = Dict{Symbol, Any}(
        :parallel => true,  
        :walltime => 16.0 * walltime)
    runtime_final = @elapsed begin
        model = fit_model(
            y, ptr, brks, wts, istemp,
            λ1, λ2, η1, η2, modelopts, fitopts)
    end
    betasfile = joinpath(
        workdir(),
        betas_relpath(),
        "betas_" * filename[6:end-4] * ".csv")
    resultsfile = joinpath(
        workdir(),
        fitmetrics_relpath(),
        "cvloss_" * filename[6:end-4] * ".csv")
    gpfile = joinpath(
        workdir(),
        fitmetrics_relpath(),
        "gp_" * filename[6:end-4] * ".csv")

    # write results
    CSV.write(resultsfile, results)
    CSV.write(gpfile, gpdf)
    open(betasfile, "w") do f
        writedlm(f, model.beta, ',')
    end
    
    println("Best model:")
    println("   split:     ", filename)
    println("   params:    ", (λ1, λ2, η1, η2))
    println("   predicted: ", best_predicted)
    println("   runtime:   ", runtime_final)
    println("   generation:   ", results.gen[bestidx])
    steps = model.steps
    conv = model.converged
    t = "$(Dates.Time(Dates.now()))"[1:5]
    pnorm = model.prim_norms[end]
    dnorm = model.dual_norms[end]
    loss = model.loss[end]
    admm_penalty = model.admm_penalty
    println("steps=$steps, conv=$conv, t=$t, loss=$loss, pnorm=$pnorm, dnorm=$dnorm, alpha=$admm_penalty")
end


# define and call main routine using distributed computing
function main(data_targets, walltime, ngens, gensize, nsplits)
    @sync @distributed for filename in data_targets
        fit_split(filename, walltime, nsplits, ngens, gensize)
    end
end


main(data_targets, walltime, ngens, gensize, nsplits)


# clean up processes
for p in workers()
    rmprocs(p)
end
