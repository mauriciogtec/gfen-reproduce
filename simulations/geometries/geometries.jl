using Pkg; Pkg.activate("GraphFusedElasticNet.jl")
using LinearAlgebra
using Distributions, StatsFuns
using Plots, Images
using Random
using YAML, DotMaps
using GraphFusedElasticNet


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
            successes[i, j] = rand(Bernoulli(p))
        end
    end
end

##
plot(Gray.(1.0 .- signal))


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
u = 1.0 .- successes ./ (attempts .+ 1e-6)
z = [
    miss[i, j] ? RGB(1., 0., 0.) : RGB(u[i, j], u[i, j], u[i, j])
    for i in 1:nr, j in 1:nc
]

plt = plot(z)
xlabel!("time")
ylabel!("space")

##


# spatial neighbor correlation
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

#
X = [x tnb snb dnb]
cor(X)

# now the smoothing
ptr, brks = grid_trails(nr, nc)

# MAP estimate
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
mod = BinomialGFEN(ptr, brks, modelopts...)
