using Pkg; Pkg.activate("GraphFusedElasticNet.jl")
using Distributed


# only set to true if running interactive
# not if running from batch file
debug_mode = false  # Base.isinteractive()
if debug_mode
    walltime = 10.0
    ngens = 8
    gensize = 4
    cvsplits = 5
    data_targets = readdir("productivity_splits")
    num_procs = 1
else
    walltime = parse(Float64, ARGS[1])
    ngens = parse(Int, ARGS[2])
    gensize = parse(Int, ARGS[3])
    cvsplits = parse(Int, ARGS[4])
    data_targets = ARGS[5:end]
    num_procs = 1
end
if num_procs > 1
    addprocs(num_procs - 1, exeflags="--project=GraphFusedElasticNet.jl")
end

# relevant path (ADAPT TO CLUSTER IF NECESSARY)
@everywhere begin
    # instantiate and load libraries in every process, define globals
    using JSON
    using DataFrames
    using Dates
    using Distributions
    using Random
    using CSV
    using Base.Threads
    using DelimitedFiles
    using StatsFuns
    using GraphFusedElasticNet
    
    workdir() = pwd()
    traildata_relpath() = "processed_data/trails.json"
    betas_relpath() = "best_betas/"
    fitmetrics_relpath() = "modelfit_metrics/"
    data_relpath() = "productivity_splits/"
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
@everywhere function generate_cvsets(y, cvsplits)
    # make the cv splits
    # the only trick is not to count the integers that 
    # have missing data
    attempts = y[:, 1]
    N = length(attempts)
    cvsets = [Set{Int}() for i in 1:cvsplits]
    iobs = [i for (i, a) in enumerate(attempts) if a >= 1.0]
    shuffle!(iobs)
    Nobs = length(iobs)
    splitsize = Nobs ÷ cvsplits
    for k in 1:cvsplits
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


# fits a binomial GFEN model for given params and data
@everywhere function fit_model(
    ytrain, ptr, brks, wts, istemp, λ1, λ2, η1, η2, modelopts, fitopts; ϵ=1e-5
)
    N = size(ytrain, 1)
    lambdasl1 = [w * (t ? η1 : λ1) for (t, w) in zip(istemp, wts)]
    lambdasl2 = [w * (t ? η2 : λ2) for (t, w) in zip(istemp, wts)]
    # define and train model
    model = BinomialGFEN(ptr, brks, lambdasl1, lambdasl2; modelopts...)
    succ = ytrain[:, 1] .+ ϵ
    att = ytrain[:, 2] .+ 2ϵ
    fit!(model, succ, att; fitopts...)
    return model
end

                
# finds best hyperparams using cross validation
@everywhere function cv_eval(
    y, ptr, brks, wts, istemp, pars, cvsets, modelopts, fitopts, usethreads
)
    # for each cv split get the mse error
    N = size(y, 1)
    cvsplits = length(cvsets)
    npars = length(pars)
    cv_logll = zeros(npars)
    thrdid = zeros(Int, npars)   
    avtime = zeros(npars)      
    avalpha = zeros(npars)      
    # prepare the tree structure and the bint 
    Nobs = sum(y[:, 2])
    Threads.@threads for k = 1:npars
        λ1, λ2, η1, η2 = pars[k]
        for i = 1:cvsplits
            thrdid[k] = threadid()
            # get the cv vector with missing data
            ytrain = get_train_data(y, cvsets[i])
            runningtime = @elapsed begin
                model = fit_model(
                    ytrain, ptr, brks, wts, istemp,
                    λ1, λ2, η1, η2, modelopts, fitopts)
            end
            beta = model.beta
            avtime[k] += runningtime   
            avalpha[k] += model.admm_penalty
            # compute the out-of-sample likelihood
            ll = 0.0
            for j in cvsets[i]
                s, N = y[j, :]
                β = beta[j]
                if beta[j] >= 0
                    ll += - N * log(1.0 + exp(-β)) - (N - s) * β
                else
                    ll += s * β - N * log(1.0 + exp(β))
                end
            end
            cv_logll[k] = ll 
        end
    end
    cv_logll /= Nobs
    avtime /= cvsplits
    avalpha /= cvsplits
    cv_logll, thrdid, avtime, avalpha
end


# main program
@everywhere function fit_split(filename, walltime, cvsplits, ngens, gensize, eval_corners=true)
    # read trail data (it is then copied to each worker by pmap)
    ptr, brks, wts, istemp, num_nodes = loadtrails()

    # set up gaussian process with smoothing parameters
    # (7 ^ 4) = 2401 parameter space size
    # Kernel matrix is 2401 x 2401 (196,882 entries)
    # for much larger spaces consider more efficient
    # implementations of gaussian processes
    sl1 = Uniform(-4.0, 1.0)
    sl2 = Uniform(-4.0, 2.0)
    tl1 = Uniform(-4.0, 1.0)
    tl2 = Uniform(-4.0, 2.0)
    dists = [sl1, sl2, tl1, tl2]

    a, σ, b = 0.1, 0.00001, 0.0001^2
    gp = RandomGaussianProcessSampler(dists, a=a, σ=σ, b=b)
    gp_offset = 0.0  # empirically assigned to running obs mean

    # model parameters
    modelopts = Dict{Symbol, Any}(
        :admm_balance_every => 10,
        :admm_init_penalty => 0.1,
        :admm_residual_balancing => true,
        :admm_adaptive_inflation => true,
        :reltol => 5e-3,
        :admm_min_penalty => 0.1,
        :admm_max_penalty => 5.0,
        :abstol => 1e-5,
        :save_norms => true,
        :save_loss => true
    )
    fitopts = Dict{Symbol, Any}(
        :walltime => walltime,
        :parallel => false
    )

    # read data
    fname = joinpath(workdir(), data_relpath(), filename)
    y = readdlm(fname, ',') 
    cvsets = generate_cvsets(y, cvsplits)  
    
    # df = Vector{DataFrame}(undef, ngens)
    results = []
    corners = [
        [a, b, c, d]
        for a in (sl1.a, sl1.b)
        for b in (sl2.a, sl2.b)
        for c in (tl1.a, tl1.b)
        for d in (tl2.a, tl2.b)
    ]
    for gen in 1:ngens
        # sample lambdas from generation
        hparams, pred, band = gpsample(gp, gensize)
        while length(corners) > 0
            for j in 1:min(gensize, length(corners))
                hparams[:, j] = pop!(corners)
                pred[j] = -Inf
                band[j] = 0
            end
        end

        # run for every lambda in thread
        fns = fill(filename, gensize)
        pids = fill(getpid(), gensize)
        hosts = fill(gethostname()[1:8], gensize)
        gens = fill(gen, gensize)
        
        # obtain cv losses and update gaussian process
        usethreads = true
        pars = [10 .^ hparams[:, i] for i in 1:gensize]
        cv_logll, thrds, avtime, avalpha = cv_eval(
            y, ptr, brks, wts, istemp,
            pars, cvsets,
            modelopts, fitopts, usethreads
        )
        for (xi, yi) in zip(pars, cv_logll)
            addobs!(gp, xi, yi)
        end
        # update the offset helps with better estimates
        # at far regions from observed data
        # it's like using an empirical prior, it doesn't 
        # affect much regions with densely observed data
        N0, N1 = (gen - 1) * gensize, gen * gensize
        gp_offset = (gp_offset *  N0 +  sum(cv_logll)) / N1
        gp.offset = gp_offset

        pars = hcat(pars...)
        slambdasl1 = pars[1, :]
        slambdasl2 = pars[2, :]
        tlambdasl1 = pars[3, :]
        tlambdasl2 = pars[4, :]
        df = DataFrame(
            fn=fns, pid=pids, host=hosts, thread=thrds,
            λsl1=slambdasl1, λsl2=slambdasl2, λtl1=tlambdasl1, λtl2=tlambdasl2,
            cv_logll=cv_logll, time=avtime, alpha=avalpha, gen=gens,
            prev_pred=pred, prev_sd=band
        )
        println(df)
        push!(results, df)
    end

    # now obtain the best hyperperameters and train the best model
    results = vcat(results...)
    df = results
    X = [[x...] for x in zip(df.slambdasl1, df.slambdasl2, df.tlambdasl1, df.tlambdasl2)]
    pred, band = gpeval(gp, X)
    results.final_pred = pred
    results.final_sd = band
    #
    bestidx = argmax(results.final_pred)
    best_predicted = results.final_pred[bestidx]
    λ1 = df.slambdasl1[bestidx]
    λ2 = df.slambdasl2[bestidx]
    η1 = df.tlambdasl1[bestidx]
    η2 = df.tlambdasl2[bestidx]

    gpdf = DataFrame(
        slambdasl1=df.slambdasl1,
        slambdasl2=df.slambdasl2,
        tlambdasl1=df.tlambdasl1,
        tlambdasl2=df.tlambdasl2,
        pred=pred,
        band=band
    )
    println("all tested points:")
    println(gpdf)

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
        "betas_" * filename
    )
    resultsfile = joinpath(
        workdir(),
        fitmetrics_relpath(),
        "cvloss_" * filename
    )
    gpfile = joinpath(
        workdir(),
        fitmetrics_relpath(),
        "gp_" * filename
    )

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
function main(data_targets, walltime, ngens, gensize, cvsplits)
    # if in slurm cluster environment (TACC)
    local_rank = haskey(ENV, "SLURM_PROCID") ? parse(Int, ENV["SLURM_PROCID"]) : -1

    if local_rank >= 0
        # running in MPI style of parallel program, each program has
        #   a rank and selects one target
        data_targets = [data_targets[local_rank + 1]]
    end

    if num_procs > 1
        @sync @distributed for filename in data_targets
            # running in in-program distributed mode, the program will
            #   schedule a file for each subprocess
            m = "Processing $(filename) in process $(getpid()) with rank $(local_rank)..."
            println(m)
            fit_split(filename, walltime, cvsplits, ngens, gensize)
        end
    else
        for filename in data_targets
            m = "Processing $(filename) in process $(getpid()) with rank $(local_rank)..."
            println(m)
            fit_split(filename, walltime, cvsplits, ngens, gensize)
        end
    end
end


main(data_targets, walltime, ngens, gensize, cvsplits)
