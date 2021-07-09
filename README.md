# How Likely are Ride-share Drivers to Earn a Living Wage? Large-scale Spatio-temporal Density Smoothing with the Graph-fused Elastic Net

This repository contains all the code to necessary to reproduce our results. Here's a brief description of the contents of the repository. See the file `MANIFEST.md` for a description of software versions used.

For completeness, all the scripts for preprocessing the raw RideAustind ata are included. To run those, it is necessary to have `R` and `python 3`. For convenience, the post-processed data are also included, so pre-processing steps can be skipped and start at step 2 below.

The main algorithms are implemented using the high-performance computing software `Julia (v1.5)` with an open-source package we developed for this paper called `GraphFusedElasticNet.jl`. Below we provide further instructions.


## 0. Julia Package Installation

All the algorithms to run the GFEN are available in our Julia package `GraphFusedElasticNet.jl`. To install the package from the Internet you can use the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("https://github.com/mauriciogtec/GraphFusedElasticNet.jl")
```

In the future, after we test it in a few other problems, we will register this package as an official Julia package. As an alternative to direct installation from the Internet into the base system, the package can be loaded directly from the local source code. More specifically, the package is configured as a git submodule of this Github repository. To clone the repository with the submodule activated one must execute the following command when cloning this repo:

```bash
git clone --recurse-submodules https://github.com/mauriciogtec/gfen-reproduce
```

All the Julia scripts that we describe below assume that the package source is locally available as when using the above command. To load the package locally in Julia one can run

```julia
using Pkg
Pkg.add("https://github.com/mauriciogtec/GraphFusedElasticNet.jl")
```

```julia
Pkg.activate("./GraphFusedElasticNet.jl/")
Pkg.instantiate()
```

This will create a local environment to run the code that guarantees the right dependencies. Remove these lines if you want to run the scripts from a package installation from the Internet.

 
## 1. Pre-processing Raw Data

### 1.1 Ride Austin Data

Our data consists of approximately 1.4 million trips from RideAustin. The raw database can be found at [`data.world`](https://data.world/ride-austin). For convenience, it can also be found in this repository as we used in in our analysis in the folder `raw_data/`. We also use geographical data from Traffic Analysis Zones (TAZs) in the city of Austin. The shapefiles for the TAZs are in the folder `raw_data/shapefiles/`.


### 1.2. Splits and Graph

We clean the data and obtain a measure of the productivity in $/hours as explained in our paper for each trip. This is the expected productivity given the location of a trip. The cleaned database is in the folder `processed_data/rideaustin_productivity`. This dataset also contains other generated quantities (e.g., trip start and end coordinates, idle time, reach time, duration of subsequent trip) which are used to compute the productivity measure.

To reproduce the cleaning steps it is necessary to run the R scripts `1_process_data_and_adjacency.R` and `1b_create_binary_splits.R`. These scripts also uses the TAZ polygon information to find adjacent TAZs, the adjacency information is in the file `processed_data/taz_adjacency.csv`. Finally, it saves the quantiles of the all-combined empirical distribution of productivity that will be used a splitting values for our binary tree density estimation approach and saves them in the file `processed_data/splits_qua.csv`.

To perform spatiotemporal smoothing we need a spatiotemporal graph that contains edges from temporal and spatial adjacency. The graph edges are generated in the Python script `2_process_graph_and_splits.py` with approx. 200k nodes and 950k edges. The vertices of the graph are enumerated and information about vertex and space-TAZ correspondence is stored in the file `processed_data/vertex_data.csv`. This Python script also partitions the productivity data using the split levels generated in the preprocessing step. The resulting splits are stored in the folder `productivity_splits/`. There is exactly one file per split, for which we will training a Binomial model. The format of the data is the following, corresponding to a split `(a, b)`, each row contains to entries `(s_i, n_i)` where `n_i` are the number of counts occurring in the range `(a, b)` and `s_i` are the number of occurrences in the range `(a, (a + b) / 2)`. See our paper for a longer description.

### 1.3 Smoothing trails (pre-processing)

For our scalable algorithm we need to decompose the graph into trails. These trails are the basis for scalable performance since can be smoothed in linear time. The script that generates the trails is `3_generate_trails.jl`. It will produce the file `processed_data/trails.json` that contains the resulting trails. It contains for objects: a _pointer_ array that shows the nodes visited sequentially by the trail, note that all the trails are concatenated; a _breaks_ array indicates the beginning index and end index of each trail in the pointer; a _temporal indicator_ array indicates if the edge in-between visited vertices in the order specified by the pointer is temporal or spatial (used to assign different hyper-parameters); finally, since the algorithm can be run with a trail partition that visits each node more than once, there is a _weights_ array inversely proportional to the number of visits.

## 2. Training

### 2.1 Hyperparameter search

The file `4_hyperparameter_search.jl` executes a Graph Fused Elastic-Net Binomial model at each split level. Each time it is executed it will produce a file with model fit metrics such as cross-validation error and the fitted Gaussian process for the hyper-parameters at the folder `modelfit_metrics`.

<!-- It will use the best hyper-parameters to fit a model and the output is stored at `best_betas`. These betas are log-odds and there's exactly one for each split. Each split can be optimized in parallel. Once the optimization for every split is completed, the script `6_densities_from_fitted_bettas.jl` can be used to compute the actual probabilities associated to each bin of the tree. The result will be a matrix of size `M x V` where `M` is the number of bins (determined by the number of splits) and `V` is the number of vertices in the graph. The output is stored at `output_smooth_probs/`. This script has the option of storing the results into chunks, due to the size of the matrix. -->


We design our scripts having a cluster environment in mind. The ideas is that we can use multiple levels of parallelism. For example, to fit 32 splits we can divide them into 8 jobs, each one running at a different node. Each node can launch 8 distributed processes, one per split. Each processes can have several cores/threads assigned and used multi-threading to test many hyper-parameters in parallel, for example, using 16 threads depending on the environment. With this design, the task that will typically take about a week can be run in below an hour. We fitted our models using the system Stampede 2 at the [Texas Advanced Computing Center (TACC)](https://www.tacc.utexas.edu). This system is based on SLURM, a common cluster manager. We provide a script `4b_hyperparameter_search_slurmjobs.jl` that can be parametrized (see the first lines of the file) and will produce `bash` files in the `job_scripts` folder that can be used in the super-computing environment.

Hyper-parameter tuning uses the fast optimization-based algorithm described in the paper paired with Bayesian optimization. However, it can take several hours or days to run even in good computing environments. Since the `modelfit_metrics/` folder already contains results from the hyper-parameter optimiation, this step can be skipped and go to 3.3.

### 2.2 Model fit

The script `5_modelfit_script.jl` takes the best parameters stored in `modelfit_metrics/` and samples from a fully Bayesian using MCMC as described in the paper with warm starts from the MAP estimates. 

Once the model for all splits finishes running (about 6-12 hours per tree split). The densities can be calculated using the script `6_densities_from_fitted_betas.jl`. This script will create arrays of logits (saved in `.npy` format) and store them in the `fitted_densities` folder. The logits are evaluated for the all the splits of lower level. For example, if the tree is balanced with depth `d`, this amounts to `2^d - 1` splits.

The repository contains the results from MAP estimates for all nodes, but only contains the full MCMC results for a handful of example representative locations (discussed in the paper) due to size constraints.

## 3. Paper Figures

Most of the figures in the paper can be reproduced from the data contained in the repository. See the `R` notebook `7_paper_plots.Rmd`.

## 4. Shiny App

Some of the results can be visualized at our [Shiny App](https://mauriciogtec.shinyapps.io/gfen/). For convenience, the source code of the app is also a submodule of this repository.


## Further Questions

Please email Mauricio Tec at `mauriciogtec@utexas.edu` for any remaining questions.

Bon voyage!







![smoothing-animation](./map_animation.gif)
