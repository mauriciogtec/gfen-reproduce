using Pkg; Pkg.activate("GraphFusedElasticNet.jl")
using LinearAlgebra
using Distributions, StatsFuns
using Plots, Images
using Random
using YAML, DotMaps
using GraphFusedElasticNet
using NPZ


## read config
cfg = DotMap(YAML.load_file("simulations/geometries/conf.yaml"))
Random.seed!(cfg.seed)

## cell types

nc, nr = cfg.grid.size
signal = zeros(nr, nc)
attempts = zeros(nr, nc)
successes = zeros(nr, nc)
miss = zeros(Bool, nr, nc)

## background
p = rand(Uniform(cfg.bg.prob...))
N = cfg.bg.obs
signal .= p
attempts .= N
for j in 1:nc, i in 1:nr
    successes[i, j] = rand(Binomial(N, p))
end

## triangles
N = cfg.signal.obs
for k in 1:cfg.signal.triangle.n
    a, b = cfg.signal.triangle.width
    width = rand(DiscreteUniform(a, b))
    a, b = cfg.signal.triangle.height
    height = rand(DiscreteUniform(a, b))
    # rel_height = rand(Uniform(cfg.signal.triangle.rel_height...))
    # height = ceil(Int, width * rel_height)
    c0 = rand(DiscreteUniform(1, nc - width))
    r0 = rand(DiscreteUniform(1, nr - height))
    p = rand(Uniform(cfg.signal.prob...))
    for j in c0:(c0 + width), i in r0:(r0 + height)
        m = 0.5 * height / width
        b = r0 + 0.5 * height
        if abs(i - b) ≤ m * (j - c0)
            signal[i, j] = p
            attempts[i, j] = 1
            successes[i, j] = rand(Binomial(N, p))
        end
    end
end

##
plot(Gray.(1.0 .- signal))
xlabel!("time")
ylabel!("space")

##
savefig("simulations/geometries/geometries_truth.pdf")


## ellipses
for k in 1:cfg.miss.ellipse.n
    width = rand(DiscreteUniform(cfg.miss.ellipse.width...))
    height = rand(DiscreteUniform(cfg.miss.ellipse.height...))
    r0 = rand(DiscreteUniform(1, nr - width))
    c0 = rand(DiscreteUniform(1, nc - height))
    miss_prob = rand(Uniform(cfg.miss.prob...))
    θ = 2π * rand(Uniform(cfg.miss.ellipse.angle...))
    R = [cos(θ) sin(θ); sin(θ) -cos(θ)]
    for j in 1:nc, i in 1:nr
        point = R * [i - r0, j - c0]
        if (point[1] / height)^2 + (point[2] / width)^2 ≤ 1.0
            j
            if rand() < miss_prob
                attempts[i, j] = 0
                successes[i, j] = 0
                miss[i, j] = true
            end
        end
    end
end

## visualize
u = 1.0 .- successes ./ (attempts .+ 1e-6);
z = [
    miss[i, j] ? colorant"#FF9AA2" : RGB(u[i,j], u[i, j], u[i,j])
    for i in 1:nr, j in 1:nc
];

plt = plot(z)
xlabel!("time")
ylabel!("space")

##
savefig("simulations/geometries/geometries_data.pdf")


## spatial neighbor correlation
x = Float64[]
tnb = Float64[]
snb = Float64[]
dnb = Float64[]
for j in 2:nc, i in 2:nr
    push!(x, signal[i, j])
    push!(tnb, signal[i, j - 1])
    push!(snb, signal[i - 1, j])
    push!(dnb, signal[i - 1, j - 1])
end

##
X = [x tnb snb dnb]
cor(X)

## now the smoothing
ptr, brks = grid_trails(nr, nc)
λ1 = 0.75
λ2 = 1.5
lambdasl1 = fill(λ1, size(ptr))
lambdasl2 = fill(λ2, size(ptr))

## MAP estimate
modelopts = Dict{Symbol, Any}(
    :admm_balance_every => 50,
    :admm_init_penalty => 10.0,
    :admm_residual_balancing => true,
    :admm_adaptive_inflation => false,
    :reltol => 1e-3,
    :admm_min_penalty => 0.1,
    :admm_max_penalty => 100.0,
    :abstol => 1e-8,
    :save_norms => true,
    :save_loss => true
)

fitopts = Dict{Symbol, Any}(
    :walltime => 300.0,
    :parallel => true
)

model = BinomialGFEN(ptr, brks, lambdasl1, lambdasl2; modelopts...)

ϵ = 1e-8
succ = vec(successes) .+ ϵ
att = vec(attempts) .+ 2ϵ
@time fit!(model, succ, att; fitopts...)

##
plot(log10.(model.prim_norms))

##
plot(log10.(model.dual_norms))


##
ω = 1. .- logistic.(model.beta)
ω = reshape(ω, nr, nc)
# npzwrite("simulations/geometries/map.npy", Float32.(ω))


# ω = Float64.(npzread("simulations/geometries/map.npy"))
plot(Gray.(ω))
xlabel!("time")
ylabel!("space")

##

# savefig("simulations/geometries/solution_lasso_1-0.pdf")
# savefig("simulations/geometries/solution_ridge_2-0.pdf")
savefig("simulations/geometries/solution_enet_0-75_1-5.pdf")


## uncertainty quantification
edges = Tuple{Int, Int}[]
λ1 = 0.75
λ2 = 1.5
idx = LinearIndices((nr, nc))
for j in 1:nc, i in 1:nr
    (i > 1) && push!(edges, (idx[i, j], idx[i - 1, j]))
    (j > 1) && push!(edges, (idx[i, j], idx[i, j - 1]))
    (i < nr) && push!(edges, (idx[i, j], idx[i + 1, j]))
    (j < nc) && push!(edges, (idx[i, j], idx[i, j + 1]))
end
bmod = BayesianBinomialGFEN(edges, tv1=λ1, tv2=λ2)


##
n = 3000
thinning = 5
burnin = 2000

##
init = model.beta  # zeros(size(mcmc_init))
@time chain = sample_chain(bmod, succ, att, n, burnin=burnin, thinning=thinning, init=init, verbose=true, async=true)

##
z = logistic.(chain)
q75 = mapslices(x -> quantile(x, 0.75), z, dims=2)
q50 = mapslices(x -> quantile(x, 0.50), z, dims=2)
q25 = mapslices(x -> quantile(x, 0.25), z, dims=2)
iqr = q75 .- q25

unc = reshape(iqr, (nr, nc))
unc = (unc .- minimum(unc)) ./ (maximum(unc) .- minimum(unc))
npzwrite("simulations/geometries/bayes_iqr.npy", Float32.(unc))
unc = Float64.(npzread("simulations/geometries/bayes_iqr.npy"))
##

plot(Gray.(1 .- unc))
xlabel!("time")
ylabel!("space")

## posterior covariance  How to order points?

## posterior inverse covariance

##
savefig("simulations/geometries/geometries_iqr.pdf")
##

median_signal = reshape(q50, (nr, nc))
npzwrite("simulations/geometries/bayes_median.npy", Float32.(median_signal))
median_signal = Float64.(npzread("simulations/geometries/bayes_median.npy"))
plot(Gray.(median_signal))
xlabel!("time")
ylabel!("space")

## now with shrinking

##
ϵ = 0.1
succ = vec(successes) .+ ϵ
att = vec(attempts) .+ 2ϵ
rmodel = BinomialGFEN(ptr, brks, lambdasl1, zeros(size(lambdasl2)); modelopts...)
fit!(rmodel, succ, att; fitopts...)
ω = logistic.(rmodel.beta)
ω = reshape(ω, nr, nc)

plot(Gray.(1 .- ω))
xlabel!("time")
ylabel!("space")

##