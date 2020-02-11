# set-up distributed environment
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
using Pkg
Pkg.activate(joinpath(workdir(), GFEN_relpath()))
Pkg.instantiate()
@everywhere begin
    using Pkg
    Pkg.activate(joinpath(workdir(), GFEN_relpath()))
    # Pkg.instantiate()
    using JSON
    using DataFrames
    using Dates
    using Random
    using CSV
    using Base.Threads
    using DelimitedFiles
    using GraphFusedElasticNet
    # include(joinpath(GFEN_relpath(), "src/GraphFusedElasticNet.jl"))
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
@everywhere function generate_cvsets(y, nsplits)
    # make the cv splits
    # the only trick is not to count the integers that 
    # have missing data
    attempts = y[:, 1]
    N = length(attempts)
    cvsets = [Set{Int}() for i in 1:nsplits]
    iobs = [i for (i, a) in enumerate(attempts) if a >= 1.0]
    shuffle!(iobs)
    Nobs = length(iobs)
    splitsize = Nobs ÷ nsplits
    for k in 1:nsplits
        for i in ((k - 1) * splitsize + 1):(k * splitsize)
            push!(cvsets[k], iobs[i])
        end
    end
    return cvsets
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

                
# finds best hyperparams using cross validation
@everywhere function cv_eval(
        y, ptr, brks, wts, istemp,
        pars, cvsets,
        modelopts, fitopts, usethreads)
    # for each cv split get the mse error
    N = size(y, 1)
    nsplits = length(cvsets)
    npars = length(pars)
    cv_logll = zeros(npars)
    thrdid = zeros(Int, npars)   
    avtime = zeros(npars)      
    # prepare the tree structure and the bint 
    Nobs = sum(y[:, 2])
    if !usethreads
        for (k, (λ1, λ2, η1, η2)) in enumerate(pars)
            for i in 1:nsplits
                # get the cv vector with missing data
                ytrain = get_train_data(y, cvsets[i])
                runningtime = @elapsed begin
                    model = fit_model(
                        ytrain, ptr, brks, wts, istemp,
                        λ1, λ2, η1, η2, modelopts, fitopts)  
                end
                beta = model.beta
                avtime[k] += runningtime
                # compute the out-of-sample likelihood
                ll = 0.0
                for j in cvsets[i]
                    s, N = y[j, :]
                    ρ = sigmoid(beta[j])
                    ll += s * log(ρ + 1e-12) + (N - s) * log(1.0 - ρ + 1e-12)
                end
                cv_logll[k] = ll 
            end
        end
    else
        @threads for k = 1:npars
            λ1, λ2, η1, η2 = pars[k]
            for i = 1:nsplits
                thrdid = threadid()
                # get the cv vector with missing data
                ytrain = get_train_data(y, cvsets[i])
                runningtime = @elapsed begin
                    model = fit_model(
                        ytrain, ptr, brks, wts, istemp,
                        λ1, λ2, η1, η2, modelopts, fitopts)
                end
                beta = model.beta
                avtime[k] += runningtime   
                # compute the out-of-sample likelihood
                ll = 0.0
                for j in cvsets[i]
                    s, N = y[j, :]
                    ρ = sigmoid(beta[j])
                    ll += s * log(ρ + 1e-12) + (N - s) * log(1.0 - ρ + 1e-12)
                end
                cv_logll[k] = ll 
            end
        end
    end
    cv_logll /= Nobs
    avtime /= nsplits
    cv_logll, thrdid, avtime
end


# main program
@everywhere function fit_split(filename, walltime, nsplits, ngens, gensize)
    # read trail data (it is then copied to each worker by pmap)
    ptr, brks, wts, istemp, num_nodes = loadtrails()

    # set up gaussian process with smoothing parameters
    sl1 = [1e-6,  0.2, 0.4, 0.6, 0.8, 1.0, 1.25]
    tl1 = [1e-6,  0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
    sl2 = [1e-6,  0.2, 0.4, 0.6, 0.8, 1.0, 1.25]
    tl2 = [1e-6,  0.5, 1.0, 2.5, 5.0, 7.5, 10.0]

    hparams = hcat([[a, b, c, d]
                    for a in sl1
                    for b in tl1
                    for c in sl2
                    for d in tl2]...)

    nl = size(hparams, 2)
    a, σ, b = 0.5, 0.0001, 1.0
    gpsampler = GaussianProcessSampler(
        hparams, zeros(nl), a=a, σ=σ, b=b)

    # model parameters
    modelopts = Dict{Symbol, Any}(
        :admm_balance_every => 10,
        :admm_init_penalty => 5.0,
        :admm_residual_balancing => true,
        :admm_adaptive_inflation => true,
        :reltol => 1e-3,
        :admm_min_penalty => 0.01,
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
        if gen == 1 && gensize == 16
            # evaluate first in all the corners !
            idx = Int[]
            a, b = minimum(sl1), minimum(tl1)
            c, d = minimum(sl2), minimum(tl2)
            A, B = maximum(sl1), maximum(tl1)
            C, D = maximum(sl2), maximum(tl2)
            for j in 1:nl
                if (hparams[1, j] in [a, A] &&
                    hparams[2, j] in [b, B] &&
                    hparams[3, j] in [c, C] &&
                    hparams[4, j] in [d, D])
                    push!(idx, j)
                end
            end
        else
            idx, _, _  = gpsample(gpsampler, gensize)
        end
        pars = [hparams[:, i] for i in idx]
        slambdasl1 = hparams[1, idx]
        tlambdasl1 = hparams[2, idx]
        slambdasl2 = hparams[3, idx]
        tlambdasl2 = hparams[4, idx]

        # run for every lambda in thread
        fns = fill(filename, gensize)
        pids = fill(getpid(), gensize)
        hosts = fill(gethostname()[1:8], gensize)
        gens = fill(gen, gensize)

        # obtain cv losses and update gaussian process
        usethreads = true
        cv_logll, thrds, avtime = cv_eval(
            y, ptr, brks, wts, istemp,
            pars, cvsets,
            modelopts, fitopts, usethreads)
        addobs!(gpsampler, idx, cv_logll)

        df = DataFrame(
            fn=fns, pid=pids, host=hosts, thread=thrds,
            λsl1=slambdasl1, λtl1=tlambdasl1, λsl2=slambdasl2, λtl2=tlambdasl2,
            cv_logll=cv_logll, time=avtime, gen=gens)
        println(df)
        push!(results, df)
    end

    # now obtain the best hyperperameters and train the best model
    results = vcat(results...)
    bestidx = argmax(results.cv_logll)
    λ1 = results.λsl1[bestidx]
    η1 = results.λtl1[bestidx]
    λ2 = results.λsl2[bestidx]
    η2 = results.λtl2[bestidx]
     
    modelopts[:reltol] /= 10.0
    fitopts = Dict{Symbol, Any}(
        :parallel => true,
        :walltime => 2.0 * walltime)
    model = fit_model(
        y, ptr, brks, wts, istemp,
        λ1, λ2, η1, η2, modelopts, fitopts)
    betasfile = joinpath(
        workdir(),
        betas_relpath(),
        "betas_" * filename[6:end-4] * ".csv")
    resultsfile = joinpath(
        workdir(),
        fitmetrics_relpath(),
        "metrics_" * filename[6:end-4] * ".csv")

    # write results
    println("...writing best model with params ", (λ1, η1, λ2, η2))
    CSV.write(resultsfile, results)
    open(betasfile, "w") do f
        writedlm(f, model.beta, ',')
    end
    
    dfbest = results[bestidx, :]
    println("Best model:")
    println(dfbest) # for fun
    steps = model.steps
    conv = model.converged
    t = "$(Dates.Time(Dates.now()))"[1:5]
    pnorm = model.prim_norms[end]
    dnorm = model.dual_norms[end]
    loss = model.loss[end]
    println("steps=$steps, conv=$conv, t=$t, loss=$loss, pnorm=$pnorm, dnorm=$dnorm")
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
